"""Seleção dos conteúdos mais relevantes para o contexto.

1. (Opcional) Reranking com cross-encoder multilíngue: lê pergunta e avaliação
   juntas e dá um score mais preciso que a similaridade de vetores. Desligado
   por padrão (USE_RERANKER=false) para manter a execução leve em CPU.
2. Filtro de relevância mínima: descarta candidatos com similaridade densa
   abaixo de MIN_RELEVANCE (evita "encher" o contexto com ruído).
3. Deduplicação: avaliações com texto idêntico ("Recebi o produto", "Ótimo")
   repetem-se milhares de vezes; mantemos uma e contamos as repetições, para
   o contexto ter DIVERSIDADE de evidências sem perder a noção de frequência.
4. Diversidade com MMR e priorização de textos informativos (ver select_evidence).
"""

from __future__ import annotations

import numpy as np

from voc_rag.retrieval.query_processing import normalize_for_matching
from voc_rag.retrieval.retriever import Candidate


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def rerank(self, question: str, candidates: list[Candidate], texts: list[str]) -> list[Candidate]:
        if not candidates:
            return candidates
        scores = self._model.predict([(question, t) for t in texts])
        for cand, score in zip(candidates, scores):
            cand.rerank_score = float(score)
        return sorted(candidates, key=lambda c: c.rerank_score, reverse=True)


def select_evidence(
    candidates: list[Candidate],
    texts: dict[int, str],
    top_k: int,
    min_relevance: float,
    embeddings: np.ndarray | None = None,
    mmr_lambda: float = 0.7,
    min_words: int = 3,
) -> tuple[list[Candidate], dict[int, int], dict]:
    """Retorna (selecionados, nº de duplicatas por posição, estatísticas).

    1. mantém candidatos com similaridade >= min_relevance
    2. colapsa textos idênticos (contando as repetições)
    3. textos muito curtos ("insatisfeito", "ótimo") vão para o fim da fila: confirmam o tema,
       mas não explicam o PORQUÊ; usados só se faltarem evidências mais informativas
    4. MMR (Maximal Marginal Relevance): escolhe iterativamente o candidato que maximiza
       λ·relevância − (1−λ)·similaridade com os já escolhidos, para o contexto cobrir
       padrões DIFERENTES em vez de 15 variações de "atraso na entrega"
    """
    relevant = [c for c in candidates if c.dense_score >= min_relevance]
    seen: dict[str, int] = {}
    duplicates: dict[int, int] = {}
    unique: list[Candidate] = []
    for cand in relevant:
        key = normalize_for_matching(texts[cand.position]).strip(" .!")
        if key in seen:
            duplicates[seen[key]] = duplicates.get(seen[key], 0) + 1
            continue
        seen[key] = cand.position
        unique.append(cand)

    informative = [c for c in unique if len(texts[c.position].split()) >= min_words]
    short = [c for c in unique if len(texts[c.position].split()) < min_words]

    # com reranker ativo, a ordem do cross-encoder é preservada (MMR usa a similaridade densa)
    use_mmr = embeddings is not None and len(informative) > top_k and all(c.rerank_score is None for c in informative)
    if use_mmr:
        chosen = _mmr(informative, embeddings, top_k, mmr_lambda)
    else:
        chosen = informative[:top_k]
    # completa com textos curtos (mais repetidos primeiro) se faltar evidência
    short.sort(key=lambda c: duplicates.get(c.position, 0), reverse=True)
    selected = chosen + short[: max(0, top_k - len(chosen))]

    stats = {
        "candidates_above_min_relevance": len(relevant),
        "duplicates_collapsed": sum(duplicates.values()),
        "short_texts_deprioritized": len(short),
        "diversity": "mmr" if use_mmr else "none",
        "selected": len(selected),
    }
    return selected, duplicates, stats


def _mmr(cands: list[Candidate], embeddings: np.ndarray, k: int, lam: float) -> list[Candidate]:
    rel = np.array([c.dense_score for c in cands])
    vecs = embeddings[[c.position for c in cands]]
    sim = vecs @ vecs.T
    chosen: list[int] = [int(np.argmax(rel))]
    while len(chosen) < k:
        redundancy = sim[:, chosen].max(axis=1)
        score = lam * rel - (1 - lam) * redundancy
        score[chosen] = -np.inf
        chosen.append(int(np.argmax(score)))
    return [cands[i] for i in chosen]
