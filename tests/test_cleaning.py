import numpy as np
import pandas as pd

from voc_rag.ingestion.cleaning import clean_reviews, combine_title_message, delivery_status, is_informative, normalize_text


def test_normalize_text_handles_missing_and_whitespace():
    assert normalize_text(None) == ""
    assert normalize_text(np.nan) == ""
    assert normalize_text("  Chegou\r\nrápido   !!!!!!  ") == "Chegou rápido !!!"


def test_is_informative():
    assert is_informative("bom")
    assert not is_informative(".")
    assert not is_informative("10")
    assert not is_informative("!!!")


def test_combine_title_message():
    assert combine_title_message("Péssimo", "não chegou") == "Péssimo. não chegou"
    assert combine_title_message("Ótimo!", "recomendo") == "Ótimo! recomendo"
    assert combine_title_message("", "só mensagem") == "só mensagem"
    assert combine_title_message("só título", "") == "só título"
    assert combine_title_message("Recomendo", "Recomendo muito") == "Recomendo muito"


def test_delivery_status():
    ts = pd.Timestamp
    assert delivery_status(pd.Series({"order_delivered_customer_date": ts("2018-01-10"),
                                      "order_estimated_delivery_date": ts("2018-01-15"),
                                      "order_status": "delivered"})) == "no_prazo"
    assert delivery_status(pd.Series({"order_delivered_customer_date": ts("2018-01-20 10:00"),
                                      "order_estimated_delivery_date": ts("2018-01-15"),
                                      "order_status": "delivered"})) == "atrasado"
    assert delivery_status(pd.Series({"order_delivered_customer_date": pd.NaT,
                                      "order_estimated_delivery_date": ts("2018-01-15"),
                                      "order_status": "canceled"})) == "cancelado"
    assert delivery_status(pd.Series({"order_delivered_customer_date": pd.NaT,
                                      "order_estimated_delivery_date": ts("2018-01-15"),
                                      "order_status": "shipped"})) == "nao_entregue"


def test_clean_reviews_dedup_and_selection():
    base = pd.DataFrame({
        "review_id": ["a", "a", "b", "c", "d"],
        "order_id": ["o1", "o2", "o3", "o4", "o5"],
        "review_score": [1, 1, 5, 3, 4],
        "review_comment_title": [None, None, "Ótimo", None, None],
        "review_comment_message": ["não chegou", "não chegou", None, None, "."],
        "review_creation_date": pd.to_datetime(["2018-01-01"] * 5),
        "order_purchase_timestamp": pd.to_datetime(["2017-12-01", "2017-12-02", "2017-12-03", "2017-12-04", "2017-12-05"]),
        "order_status": ["delivered"] * 5,
        "order_delivered_customer_date": pd.to_datetime(["2017-12-20"] * 5),
        "order_estimated_delivery_date": pd.to_datetime(["2017-12-15"] * 5),
        "customer_state": ["SP"] * 5,
        "main_category": ["x", "x", None, "y", "z"],
        "product_categories": ["x", "x", None, "y", "z"],
        "product_categories_en": ["x", "x", None, "y", "z"],
    })
    df, report = clean_reviews(base)
    assert report.duplicated_review_ids_removed == 1
    assert report.without_text == 1  # "c"
    assert report.non_informative_text == 1  # "d" -> "."
    assert set(df["review_id"]) == {"a", "b"}
    assert df.set_index("review_id").loc["b", "text_source"] == "titulo"
    assert df.set_index("review_id").loc["b", "main_category"] == "sem_categoria"
    assert (df["delivery_status"] == "atrasado").all()
