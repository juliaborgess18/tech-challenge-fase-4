"""Construção do contexto enviado ao modelo de linguagem.

Cada avaliação vira uma linha rotulada ([E1], [E2]...) com seus metadados, para
que o modelo possa (a) citar a fonte e (b) usar nota/entrega/categoria na análise.
Um resumo estatístico da amostra (distribuição de notas, entrega, categorias)
acompanha o contexto para o modelo não "chutar" frequências.
"""

from __future__ import annotations

from collections import Counter

import pandas as pd

from voc_rag.retrieval.retriever import Candidate
from voc_rag.schemas import Evidence

DELIVERY_LABELS = {
    "no_prazo": "entregue no prazo",
    "atrasado": "entregue com atraso",
    "nao_entregue": "não entregue",
    "cancelado": "cancelado",
    "entregue": "entregue",
}


def to_evidences(
    selected: list[Candidate], documents: pd.DataFrame, duplicates: dict[int, int]
) -> list[Evidence]:
    evidences = []
    for i, cand in enumerate(selected, start=1):
        row = documents.iloc[cand.position]
        evidences.append(
            Evidence(
                evidence_id=f"E{i}",
                review_id=row["review_id"],
                order_id=row["order_id"],
                text=row["text"],
                review_score=int(row["review_score"]),
                review_date=pd.Timestamp(row["review_date"]).strftime("%Y-%m-%d"),
                year_month=row["year_month"],
                main_category=row["main_category"],
                delivery_status=row["delivery_status"],
                customer_state=row["customer_state"],
                similarity=round(cand.dense_score, 4),
                fused_score=round(cand.fused_score, 5),
                rerank_score=None if cand.rerank_score is None else round(cand.rerank_score, 4),
                similar_reviews=duplicates.get(cand.position, 0),
            )
        )
    return evidences


def format_evidence_line(ev: Evidence) -> str:
    meta = (
        f"nota {ev.review_score}/5 | {ev.year_month} | categoria: {ev.main_category} | "
        f"{DELIVERY_LABELS.get(ev.delivery_status, ev.delivery_status)}"
    )
    if ev.similar_reviews:
        meta += f" | +{ev.similar_reviews} avaliação(ões) com texto idêntico"
    return f'[{ev.evidence_id}] ({meta}) "{ev.text}"'


def summarize_sample(evidences: list[Evidence]) -> str:
    if not evidences:
        return "nenhuma avaliação recuperada"
    n = len(evidences)
    scores = Counter(e.review_score for e in evidences)
    delivery = Counter(DELIVERY_LABELS.get(e.delivery_status, e.delivery_status) for e in evidences)
    cats = Counter(e.main_category for e in evidences).most_common(5)
    return "\n".join(
        [
            f"- {n} avaliações distintas recuperadas",
            "- notas: " + ", ".join(f"{s}★: {scores[s]}" for s in sorted(scores)),
            "- entrega: " + ", ".join(f"{k}: {v}" for k, v in delivery.most_common()),
            "- categorias mais presentes: " + ", ".join(f"{k} ({v})" for k, v in cats),
        ]
    )


def build_context(evidences: list[Evidence]) -> tuple[str, str]:
    """Retorna (bloco de evidências, resumo da amostra)."""
    return "\n".join(format_evidence_line(e) for e in evidences), summarize_sample(evidences)
