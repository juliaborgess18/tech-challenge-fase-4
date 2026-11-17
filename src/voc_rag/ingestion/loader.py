"""Leitura das tabelas do Olist e junção das avaliações com os metadados de pedido.

A unidade da base de conhecimento é a avaliação (review). As demais tabelas só
entram para enriquecer cada avaliação com metadados úteis para filtragem
(nota, período, entrega, categoria, estado do cliente).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

FILES = {
    "reviews": "olist_order_reviews_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "products": "olist_products_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}

DATE_COLUMNS = {
    "reviews": ["review_creation_date", "review_answer_timestamp"],
    "orders": [
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
}


def load_raw_tables(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """Lê os CSVs necessários. Lança erro claro se algum arquivo estiver faltando."""
    raw_dir = Path(raw_dir)
    missing = [f for f in FILES.values() if not (raw_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Arquivos ausentes em {raw_dir}: {missing}. "
            "Baixe o dataset com `python scripts/download_data.py` ou copie os CSVs manualmente."
        )

    tables: dict[str, pd.DataFrame] = {}
    for name, filename in FILES.items():
        # utf-8-sig remove o BOM presente no arquivo de tradução de categorias
        df = pd.read_csv(raw_dir / filename, encoding="utf-8-sig")
        for col in DATE_COLUMNS.get(name, []):
            df[col] = pd.to_datetime(df[col], errors="coerce")
        tables[name] = df
    return tables


def _order_categories(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Agrega por pedido: categorias dos produtos, nº de itens e valor total."""
    items = tables["items"].merge(
        tables["products"][["product_id", "product_category_name"]], on="product_id", how="left"
    )
    items = items.merge(tables["category_translation"], on="product_category_name", how="left")
    items["product_category_name"] = items["product_category_name"].fillna("sem_categoria")
    items["product_category_name_english"] = items["product_category_name_english"].fillna("unknown")

    def _join_unique(values: pd.Series) -> str:
        return "|".join(sorted(set(values)))

    return items.groupby("order_id").agg(
        product_categories=("product_category_name", _join_unique),
        product_categories_en=("product_category_name_english", _join_unique),
        main_category=("product_category_name", lambda s: s.value_counts().index[0]),
        n_items=("order_item_id", "count"),
        order_value=("price", "sum"),
        freight_value=("freight_value", "sum"),
    ).reset_index()


def build_review_base(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Retorna uma linha por avaliação com os metadados do pedido associados (left join)."""
    orders = tables["orders"].merge(
        tables["customers"][["customer_id", "customer_state"]], on="customer_id", how="left"
    )
    orders = orders[
        [
            "order_id",
            "order_status",
            "order_purchase_timestamp",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
            "customer_state",
        ]
    ]
    base = tables["reviews"].merge(orders, on="order_id", how="left")
    base = base.merge(_order_categories(tables), on="order_id", how="left")
    return base
