"""Tratamento de perguntas sem evidências suficientes.

Duas barreiras, ANTES e DEPOIS da geração.

Pré-geração (determinística, não gasta tokens), avaliada nesta ordem:
  1. Filtros sem resultados                                 -> INSUFFICIENT_EVIDENCE
  2. Escopo: a pergunta se parece mais com perguntas fora do domínio
     (margem do ScopeClassifier < SCOPE_MARGIN) e não há suporte lexical
     na base                                                -> OUT_OF_SCOPE
  3. Conceito desconhecido: algum termo-chave da pergunta aparece em menos
     de MIN_TERM_DF avaliações ("criptomoedas", "bitcoin")  -> INSUFFICIENT_EVIDENCE
  4. Suporte lexical: menos de MIN_EVIDENCES avaliações (no recorte filtrado)
     contêm ao menos 60% dos termos-chave                   -> INSUFFICIENT_EVIDENCE
  5. Suporte semântico: menos de MIN_EVIDENCES candidatos com similaridade
     >= MIN_RELEVANCE                                       -> INSUFFICIENT_EVIDENCE
     (não se aplica a perguntas SEM tema sobre um recorte filtrado, como
     "O que motiva as notas 3?": aí o próprio filtro define a relevância)

Pós-geração (valida a saída do LLM):
  6. O LLM devolveu o marcador SEM_EVIDENCIAS_SUFICIENTES  -> INSUFFICIENT_EVIDENCE
  7. Citações [En] inexistentes no contexto são removidas e geram aviso
  8. Resposta sem nenhuma citação gera aviso de baixa rastreabilidade

Por que não um limiar único de similaridade? Ver docs/decisoes_tecnicas.md
(calibração): com o E5, perguntas fora do escopo chegam a ter similaridade maior
que perguntas legítimas, então combinamos sinais complementares e explicáveis.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from voc_rag.generation.prompts import INSUFFICIENT_MARKER
from voc_rag.retrieval.retriever import Candidate
from voc_rag.schemas import AnswerStatus

CITATION_PATTERN = re.compile(r"\[(E\d+)\]")

MESSAGES = {
    AnswerStatus.OUT_OF_SCOPE: (
        "Esta pergunta parece estar fora do escopo da base de conhecimento, que contém apenas "
        "avaliações de clientes sobre compras, produtos, entregas e atendimento no e-commerce. "
        "Não há avaliações que permitam fundamentar uma resposta."
    ),
    AnswerStatus.INSUFFICIENT_EVIDENCE: (
        "Não encontrei evidências suficientes nas avaliações para responder a esta pergunta com "
        "segurança. Tente reformular a pergunta, ampliar o período ou remover filtros."
    ),
}


@dataclass
class EvidenceSignals:
    """Sinais calculados pelo pipeline e avaliados pelos guardrails."""

    n_candidates: int
    scope_margin: float
    key_terms: list[str]
    unknown_terms: list[str]
    lexical_support: int  # -1 = pergunta sem termos-chave (verificação não se aplica)
    relevant_count: int
    best_similarity: float
    filters_active: bool
    slice_summary: bool = False  # pergunta sem termos-chave sobre um recorte filtrado


@dataclass
class GuardrailDecision:
    allowed: bool
    status: AnswerStatus | None
    reason: str
    signals: dict = field(default_factory=dict)
    thresholds: dict = field(default_factory=dict)

    @property
    def best_similarity(self) -> float:
        return self.signals.get("best_similarity", 0.0)

    @property
    def relevant_count(self) -> int:
        return self.signals.get("relevant_count", 0)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["status"] = self.status.value if self.status else None
        data["best_similarity"] = self.best_similarity
        data["relevant_count"] = self.relevant_count
        return data


def count_relevant(candidates: list[Candidate], min_relevance: float) -> int:
    return sum(1 for c in candidates if c.dense_score >= min_relevance)


def assess_evidence(
    signals: EvidenceSignals,
    min_evidences: int,
    scope_margin: float = 0.0,
    min_relevance: float | None = None,
) -> GuardrailDecision:
    thresholds = {"scope_margin": scope_margin, "min_evidences": min_evidences, "min_relevance": min_relevance}
    sig = asdict(signals)

    def block(status: AnswerStatus, reason: str) -> GuardrailDecision:
        return GuardrailDecision(False, status, reason, sig, thresholds)

    if signals.n_candidates == 0:
        reason = "nenhuma avaliação atende aos filtros aplicados" if signals.filters_active else "nenhum candidato recuperado"
        return block(AnswerStatus.INSUFFICIENT_EVIDENCE, reason)

    if signals.scope_margin < scope_margin and signals.lexical_support < min_evidences:
        return block(
            AnswerStatus.OUT_OF_SCOPE,
            f"pergunta mais próxima de exemplos fora do domínio (margem {signals.scope_margin:+.3f}) "
            f"e sem suporte lexical na base",
        )

    if signals.unknown_terms:
        return block(
            AnswerStatus.INSUFFICIENT_EVIDENCE,
            "termo(s) praticamente ausente(s) das avaliações: " + ", ".join(signals.unknown_terms),
        )

    if 0 <= signals.lexical_support < min_evidences:
        return block(
            AnswerStatus.INSUFFICIENT_EVIDENCE,
            f"apenas {signals.lexical_support} avaliação(ões) mencionam os termos "
            f"{signals.key_terms} no recorte (mínimo: {min_evidences})",
        )

    if signals.relevant_count < min_evidences and not signals.slice_summary:
        return block(
            AnswerStatus.INSUFFICIENT_EVIDENCE,
            f"{signals.relevant_count} avaliação(ões) com similaridade >= {min_relevance} (mínimo: {min_evidences})",
        )

    return GuardrailDecision(
        True,
        None,
        f"{signals.relevant_count} avaliações relevantes (melhor similaridade {signals.best_similarity:.3f})",
        sig,
        thresholds,
    )


def validate_answer(answer: str, valid_ids: set[str]) -> tuple[str, AnswerStatus, set[str], list[str]]:
    """Valida a resposta do LLM. Retorna (resposta, status, ids citados válidos, avisos)."""
    warnings: list[str] = []
    text = answer.strip()
    if INSUFFICIENT_MARKER in text:
        return MESSAGES[AnswerStatus.INSUFFICIENT_EVIDENCE], AnswerStatus.INSUFFICIENT_EVIDENCE, set(), [
            "o modelo indicou que as evidências não sustentam uma resposta"
        ]

    cited = set(CITATION_PATTERN.findall(text))
    invalid = cited - valid_ids
    if invalid:
        warnings.append(f"citações inexistentes removidas: {sorted(invalid)}")
        for cid in invalid:
            text = text.replace(f"[{cid}]", "")
    valid = cited & valid_ids
    if not valid:
        warnings.append("a resposta não cita nenhuma evidência; verifique a fundamentação")
    return text, AnswerStatus.ANSWERED, valid, warnings
