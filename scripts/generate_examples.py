"""Gera docs/exemplos.md a partir de execuções reais do pipeline.

Uso:
    python scripts/generate_examples.py            # só recuperação (sem LLM)
    python scripts/generate_examples.py --with-llm # inclui a resposta gerada (requer LLM configurado)
"""

from __future__ import annotations

import argparse
import sys
from datetime import date

from voc_rag.config import PROJECT_ROOT
from voc_rag.generation.context_builder import DELIVERY_LABELS
from voc_rag.pipeline import VoCPipeline
from voc_rag.schemas import AnswerStatus

ANSWERED = [
    ("Investigação de reclamações", "Quais são os principais problemas relatados pelos clientes em suas avaliações?"),
    ("Tema específico", "O que os clientes relatam sobre problemas relacionados à entrega?"),
    ("Experiências positivas", "Quais aspectos da experiência de compra são mais elogiados pelos clientes?"),
    ("Investigação de insatisfação", "Quais padrões podem ser identificados nas avaliações de clientes insatisfeitos?"),
    ("Exploração baseada em evidências",
     "Quais avaliações sustentam a conclusão de que existem problemas recorrentes relacionados ao prazo de entrega?"),
    ("Metadados: categoria + sentimento", "O que os clientes mais elogiam nos produtos de cama, mesa e banho?"),
    ("Metadados: entrega atrasada (derivada das datas)",
     "Quais temas aparecem nas avaliações de pedidos com entrega atrasada?"),
    ("Metadados: categoria + período", "Quais reclamações aparecem sobre celulares em 2018?"),
]

REFUSED = [
    "Qual é a capital da Austrália?",
    "Me passe uma receita de bolo de chocolate.",
    "O que os clientes acham de pagar com criptomoedas na loja?",
    "O que os clientes acham da entrega por drone?",
    "O que os clientes reclamam sobre seguros e serviços em 2016?",
]

SHOW = 8


def fmt(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-llm", action="store_true")
    args = parser.parse_args()

    pipeline = VoCPipeline.from_settings()
    out = [
        "# Exemplos de perguntas e respostas",
        "",
        f"Saídas **reais** do pipeline sobre a base completa ({fmt(len(pipeline.store))} avaliações), "
        "geradas por `python scripts/generate_examples.py`.",
        "",
    ]
    if args.with_llm:
        out += [
            f"> As respostas foram geradas pelo Google Gemini (`{pipeline.llm.model}`). Para cada pergunta aparecem as "
            "etapas do pipeline (filtros extraídos, núcleo da pergunta, decisão do guardrail), a resposta gerada e as "
            "primeiras evidências do contexto. As marcadas com ✅ foram citadas na resposta. Como o LLM não é "
            "determinístico, uma nova execução pode produzir um texto diferente, sobre as mesmas evidências.",
            "",
        ]
    else:
        out += [
            "> **Sem o texto da resposta:** esta versão foi gerada sem o LLM. Abaixo estão as etapas determinísticas: "
            "filtros extraídos, núcleo da pergunta, decisão do guardrail e as evidências que formam o contexto enviado "
            "ao modelo. Rode `python scripts/generate_examples.py --with-llm` para incluir as respostas geradas.",
            "",
        ]
    out += [f"<!-- gerado em {date.today().isoformat()} -->", "", "## Perguntas respondidas", ""]

    for title, question in ANSWERED:
        r = pipeline.ask(question, generate=args.with_llm)
        # falha temporária do LLM (ex.: 503) cai em "somente recuperação": tenta de novo
        for _ in range(2):
            if not (args.with_llm and r.status == AnswerStatus.RETRIEVAL_ONLY):
                break
            print(f"  LLM falhou em '{title}' ({r.answer[:80]}...), tentando de novo")
            r = pipeline.ask(question, generate=True)
        if args.with_llm and r.status == AnswerStatus.RETRIEVAL_ONLY:
            print(f"  AVISO: '{title}' ficou sem resposta gerada")
        t = r.trace
        sel = t.selection_stats
        out += [
            f"### {title}",
            "",
            f"**Pergunta:** {question}",
            "",
            f"- **Filtros aplicados:** {t.filters_applied}",
            f"- **Núcleo usado na busca semântica:** `{t.processed_query['embedding_text']}`",
            f"- **Guardrail:** {t.guardrail['reason']}",
            f"- **Recuperação:** {fmt(t.retrieval_stats['documents_after_filters'])} avaliações no recorte → "
            f"{sel['candidates_above_min_relevance']} candidatas relevantes → {len(r.evidences)} evidências "
            f"distintas ({sel['duplicates_collapsed']} textos idênticos colapsados, diversidade: {sel['diversity']})",
            "",
        ]
        if r.status == AnswerStatus.ANSWERED:
            out += ["**Resposta gerada:**", "", *[f"> {line}" for line in r.answer.splitlines()], ""]
        out += ["| id | nota | entrega | categoria | avaliação |", "|---|---|---|---|---|"]
        for e in r.evidences[:SHOW]:
            dup = f" *(+{e.similar_reviews} idênticas)*" if e.similar_reviews else ""
            cited = " ✅" if e.cited else ""
            out.append(
                f"| {e.evidence_id}{cited} | {e.review_score}★ | {DELIVERY_LABELS.get(e.delivery_status, e.delivery_status)} "
                f"| {e.main_category} | {e.text.replace('|', '/')}{dup} |"
            )
        if len(r.evidences) > SHOW:
            out.append(f"| … | | | | *mais {len(r.evidences) - SHOW} evidências no contexto* |")
        out.append("")

    out += [
        "## Perguntas recusadas (fora do escopo ou sem evidências)",
        "",
        "| Pergunta | Status | Motivo (trace) |",
        "|---|---|---|",
    ]
    for question in REFUSED:
        r = pipeline.ask(question, generate=False)
        out.append(f"| {question} | `{r.status.value}` | {r.trace.guardrail['reason']} |")
    out += [
        "",
        "Nenhuma dessas perguntas chega ao LLM: a recusa é decidida antes da geração, sem custo de tokens.",
        "",
    ]

    path = PROJECT_ROOT / "docs" / "exemplos.md"
    path.write_text("\n".join(out), encoding="utf-8")
    print(f"Exemplos salvos em {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
