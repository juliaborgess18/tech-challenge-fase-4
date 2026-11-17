"""CLI para fazer perguntas à base de avaliações.

Exemplos:
    python scripts/ask.py "Quais são os principais problemas relatados pelos clientes?"
    python scripts/ask.py "O que os clientes elogiam?" --categoria cama_mesa_banho
    python scripts/ask.py "Problemas de entrega" --nota-max 2 --periodo 2018-01 2018-08
    python scripts/ask.py "..." --somente-recuperacao --trace
    python scripts/ask.py "..." --json > resposta.json
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap

from voc_rag.pipeline import VoCPipeline
from voc_rag.retrieval import MetadataFilter

STATUS_LABEL = {
    "answered": "RESPONDIDA",
    "insufficient_evidence": "EVIDÊNCIAS INSUFICIENTES",
    "out_of_scope": "FORA DO ESCOPO",
    "retrieval_only": "SOMENTE RECUPERAÇÃO",
}


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pergunta")
    p.add_argument("--nota-min", type=int)
    p.add_argument("--nota-max", type=int)
    p.add_argument("--categoria", action="append", default=[], help="pode repetir")
    p.add_argument("--entrega", action="append", default=[], choices=["no_prazo", "atrasado", "nao_entregue", "cancelado"])
    p.add_argument("--periodo", nargs=2, metavar=("INICIO", "FIM"), help="YYYY-MM YYYY-MM")
    p.add_argument("--sem-filtros-automaticos", action="store_true")
    p.add_argument("--somente-recuperacao", action="store_true", help="não chama o LLM")
    p.add_argument("--top-k", type=int)
    p.add_argument("--trace", action="store_true", help="mostra o trace e o contexto enviado ao LLM")
    p.add_argument("--json", action="store_true", help="saída completa em JSON")
    args = p.parse_args()

    flt = MetadataFilter(
        min_score=args.nota_min,
        max_score=args.nota_max,
        categories=args.categoria,
        delivery_status=args.entrega,
        period_start=args.periodo[0] if args.periodo else None,
        period_end=args.periodo[1] if args.periodo else None,
    )

    pipeline = VoCPipeline.from_settings()
    resp = pipeline.ask(
        args.pergunta,
        filters=flt,
        auto_filters=not args.sem_filtros_automaticos,
        generate=not args.somente_recuperacao,
        top_k=args.top_k,
    )

    if args.json:
        print(json.dumps(resp.to_dict(), ensure_ascii=False, indent=2, default=str))
        return 0

    print(f"\nPergunta: {resp.question}")
    print(f"Status:   {STATUS_LABEL[resp.status.value]}")
    print(f"Filtros:  {resp.trace.filters_applied}")
    print("\n" + "=" * 80 + "\n" + resp.answer + "\n" + "=" * 80)
    for w in resp.warnings:
        print(f"[aviso] {w}")

    if resp.evidences:
        print(f"\nEvidências ({len(resp.evidences)}):")
        for ev in resp.evidences:
            mark = "*" if ev.cited else " "
            dup = f" (+{ev.similar_reviews} idênticas)" if ev.similar_reviews else ""
            head = f"{mark}[{ev.evidence_id}] nota {ev.review_score} | {ev.year_month} | {ev.main_category} | {ev.delivery_status} | sim={ev.similarity:.3f}{dup}"
            print(head)
            print(textwrap.indent(textwrap.fill(ev.text, 100), "      "))
            print(f"      review_id: {ev.review_id}")

    if args.trace:
        t = resp.trace
        print("\n--- TRACE ---")
        print("Transformações da pergunta:", t.processed_query.get("notes"))
        print("Recuperação:", t.retrieval_stats)
        print("Seleção:", t.selection_stats)
        print("Guardrail:", t.guardrail)
        print("Tempos (ms):", t.timings_ms)
        if t.context_preview:
            print("\n--- CONTEXTO ENVIADO AO LLM ---\n" + t.context_preview)
    return 0


if __name__ == "__main__":
    sys.exit(main())
