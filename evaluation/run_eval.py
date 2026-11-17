"""Avaliação do pipeline RAG.

Métricas (calculadas sem LLM, não gastam tokens):
  - guardrail_accuracy: % de perguntas com a decisão correta (responder vs. recusar)
  - precision@k (proxy): fração das evidências cujo texto casa com o regex do tema
  - filter_accuracy: % das perguntas com filtros esperados extraídos corretamente
  - filter_compliance: % das evidências que respeitam os filtros (deve ser 100%)

Métricas adicionais quando o LLM está configurado (--with-llm):
  - citation_rate: % das respostas com ao menos uma citação válida
  - avg_citations: média de evidências citadas por resposta
  - llm_failures: perguntas em que a chamada ao LLM falhou (caem em "somente recuperação")

Uso:
    python evaluation/run_eval.py
    python evaluation/run_eval.py --with-llm
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from voc_rag.pipeline import VoCPipeline
from voc_rag.retrieval.query_processing import normalize_for_matching
from voc_rag.schemas import AnswerStatus

EVAL_DIR = Path(__file__).resolve().parent
REFUSAL = {AnswerStatus.OUT_OF_SCOPE, AnswerStatus.INSUFFICIENT_EVIDENCE}


def check_filters(expected: dict, actual: dict) -> bool:
    for key, value in expected.items():
        got = actual.get(key)
        if isinstance(value, list):
            if not set(value).issubset(set(got or [])):
                return False
        elif got != value:
            return False
    return True


def evidence_respects(ev, flt: dict) -> bool:
    if flt.get("min_score") and ev.review_score < flt["min_score"]:
        return False
    if flt.get("max_score") and ev.review_score > flt["max_score"]:
        return False
    if flt.get("delivery_status") and ev.delivery_status not in flt["delivery_status"]:
        return False
    if flt.get("period_start") and ev.year_month < flt["period_start"]:
        return False
    if flt.get("period_end") and ev.year_month > flt["period_end"]:
        return False
    return True


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-llm", action="store_true", help="também gera respostas com o LLM")
    parser.add_argument("--questions", type=Path, default=EVAL_DIR / "questions.yaml")
    args = parser.parse_args()

    questions = yaml.safe_load(args.questions.read_text(encoding="utf-8"))["questions"]
    pipeline = VoCPipeline.from_settings()

    rows = []
    for q in questions:
        resp = pipeline.ask(q["question"], generate=args.with_llm)
        refused = resp.status in REFUSAL
        correct = refused if q["expected"] == "refuse" else not refused
        flt = resp.trace.processed_query["filters"]

        precision = None
        if q.get("relevance_regex") and resp.evidences:
            pattern = re.compile(q["relevance_regex"])
            hits = sum(bool(pattern.search(normalize_for_matching(e.text))) for e in resp.evidences)
            precision = hits / len(resp.evidences)

        row = {
            "id": q["id"],
            "expected": q["expected"],
            "status": resp.status.value,
            "correct_decision": correct,
            "best_similarity": resp.trace.guardrail.get("best_similarity"),
            "relevant_count": resp.trace.guardrail.get("relevant_count"),
            "n_evidences": len(resp.evidences),
            "precision_at_k": None if precision is None else round(precision, 3),
            "filters_ok": check_filters(q["expected_filters"], flt) if q.get("expected_filters") else None,
            "filter_compliance": (
                sum(evidence_respects(e, flt) for e in resp.evidences) / len(resp.evidences)
                if resp.evidences
                else None
            ),
            "filters": resp.trace.filters_applied,
        }
        if args.with_llm and resp.status == AnswerStatus.ANSWERED:
            row["n_citations"] = len(resp.cited_evidences)
            row["answer"] = resp.answer
        rows.append(row)
        flag = "OK " if correct else "ERR"
        sig = resp.trace.guardrail.get("signals", {})
        row["scope_margin"] = sig.get("scope_margin")
        row["lexical_support"] = sig.get("lexical_support")
        row["reason"] = resp.trace.guardrail.get("reason")
        print(f"[{flag}] {q['id']:<24} {resp.status.value:<22} margin={sig.get('scope_margin', 0):+.3f} "
              f"lex={sig.get('lexical_support')!s:>5} rel={row['relevant_count']!s:>3} p@k={row['precision_at_k']} "
              f"| {row['reason']}")

    df = pd.DataFrame(rows)
    answer_rows = df[df["expected"] == "answer"]
    summary = {
        "n_questions": len(df),
        "guardrail_accuracy": round(df["correct_decision"].mean(), 3),
        "answer_recall": round(answer_rows["correct_decision"].mean(), 3),
        "refusal_recall": round(df[df["expected"] == "refuse"]["correct_decision"].mean(), 3),
        "mean_precision_at_k": round(answer_rows["precision_at_k"].dropna().mean(), 3),
        "filter_accuracy": round(df["filters_ok"].dropna().astype(float).mean(), 3),
        "filter_compliance": round(df["filter_compliance"].dropna().mean(), 3),
    }
    if args.with_llm:
        # chamadas ao LLM que falharam (ex.: 503) caem em retrieval_only: contamos à parte
        summary["llm_failures"] = int((df["status"] == AnswerStatus.RETRIEVAL_ONLY.value).sum())
    if args.with_llm and "n_citations" in df:
        answered = df["n_citations"].dropna()
        summary["citation_rate"] = round((answered > 0).mean(), 3) if len(answered) else None
        summary["avg_citations"] = round(answered.mean(), 2) if len(answered) else None

    summary = {k: (None if isinstance(v, float) and v != v else v) for k, v in summary.items()}  # NaN -> null
    print("\nResumo:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    out_dir = EVAL_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(out_dir / f"eval_{stamp}.csv", index=False, encoding="utf-8")
    (out_dir / f"summary_{stamp}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nResultados salvos em {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
