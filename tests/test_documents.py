import pandas as pd

from voc_rag.ingestion.documents import DOCUMENT_COLUMNS, build_documents, score_to_sentiment
from voc_rag.retrieval.filters import MetadataFilter, apply_filter


def test_score_to_sentiment():
    assert [score_to_sentiment(s) for s in range(1, 6)] == ["negativa", "negativa", "neutra", "positiva", "positiva"]


def test_build_documents_columns_and_period_fallback():
    clean = pd.DataFrame({
        "review_id": ["a", "b"],
        "order_id": ["o1", "o2"],
        "text": ["não chegou", "ótimo"],
        "text_source": ["mensagem", "titulo"],
        "review_score": [1, 5],
        "review_creation_date": pd.to_datetime(["2018-02-01", "2018-03-01"]),
        "order_purchase_timestamp": pd.to_datetime(["2018-01-10", None]),
        "order_status": ["delivered", "delivered"],
        "delivery_status": ["atrasado", "no_prazo"],
        "delivery_delay_days": [3.0, -2.0],
        "main_category": ["x", "y"],
        "product_categories": ["x", "y"],
        "product_categories_en": ["x", "y"],
        "customer_state": ["SP", "RJ"],
        "n_items": [1, None],
        "order_value": [10.0, None],
    })
    docs = build_documents(clean)
    assert list(docs.columns) == DOCUMENT_COLUMNS
    assert docs["doc_id"].tolist() == [0, 1]
    assert docs.set_index("review_id").loc["a", "year_month"] == "2018-01"
    assert docs.set_index("review_id").loc["b", "year_month"] == "2018-03"  # sem compra -> data da avaliação


def test_apply_filter(documents):
    mask = apply_filter(documents, MetadataFilter(max_score=2, categories=["telefonia"]))
    assert set(documents[mask]["review_id"]) == {"r1", "r2"}
    mask = apply_filter(documents, MetadataFilter(period_start="2018-01", delivery_status=["no_prazo"]))
    assert set(documents[mask]["review_id"]) == {"r5", "r6", "r8"}
    assert apply_filter(documents, MetadataFilter()) is None


def test_filter_merge_prefers_override():
    auto = MetadataFilter(min_score=1, max_score=3, categories=["telefonia"])
    merged = auto.merge(MetadataFilter(max_score=2))
    assert (merged.min_score, merged.max_score, merged.categories) == (1, 2, ["telefonia"])
