"""Organização dos documentos da base de conhecimento.

Decisão: 1 documento = 1 avaliação (sem chunking). As avaliações são curtas
(mediana ~53 caracteres, máximo ~210), então dividir não faz sentido e manter a
avaliação inteira garante rastreabilidade direta (doc -> review_id).

O texto embedado é SOMENTE o comentário do cliente. Os metadados ficam em colunas
separadas e são usados como filtros, não misturados ao texto, para não
"contaminar" a similaridade semântica com tokens como "nota 5" ou nomes de categoria.
"""

from __future__ import annotations

import pandas as pd

DOCUMENT_COLUMNS = [
    "doc_id",
    "review_id",
    "order_id",
    "text",
    "text_source",
    "review_score",
    "sentiment",
    "review_date",
    "year_month",
    "purchase_date",
    "order_status",
    "delivery_status",
    "delivery_delay_days",
    "main_category",
    "product_categories",
    "product_categories_en",
    "customer_state",
    "n_items",
    "order_value",
]


def score_to_sentiment(score: int) -> str:
    if score <= 2:
        return "negativa"
    if score == 3:
        return "neutra"
    return "positiva"


def build_documents(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Transforma o dataframe limpo no formato final de documentos."""
    docs = pd.DataFrame(
        {
            "review_id": clean_df["review_id"],
            "order_id": clean_df["order_id"],
            "text": clean_df["text"],
            "text_source": clean_df["text_source"],
            "review_score": clean_df["review_score"].astype(int),
            "review_date": pd.to_datetime(clean_df["review_creation_date"]),
            "purchase_date": pd.to_datetime(clean_df["order_purchase_timestamp"]),
            "order_status": clean_df["order_status"],
            "delivery_status": clean_df["delivery_status"],
            "delivery_delay_days": clean_df["delivery_delay_days"],
            "main_category": clean_df["main_category"],
            "product_categories": clean_df["product_categories"],
            "product_categories_en": clean_df["product_categories_en"],
            "customer_state": clean_df["customer_state"],
            "n_items": clean_df["n_items"].fillna(0).astype(int),
            "order_value": clean_df["order_value"].fillna(0.0).round(2),
        }
    )
    docs["sentiment"] = docs["review_score"].map(score_to_sentiment)
    # Período: usa a data da compra; se ausente, a data da avaliação
    ref_date = docs["purchase_date"].fillna(docs["review_date"])
    docs["year_month"] = ref_date.dt.strftime("%Y-%m")
    docs = docs.sort_values(["review_date", "review_id"]).reset_index(drop=True)
    docs.insert(0, "doc_id", range(len(docs)))
    return docs[DOCUMENT_COLUMNS]
