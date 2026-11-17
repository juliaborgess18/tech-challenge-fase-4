"""Limpeza e preparação dos textos das avaliações + tratamento de valores ausentes."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WHITESPACE = re.compile(r"\s+")
_REPEATED_PUNCT = re.compile(r"([!?.,])\1{2,}")
_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)

MIN_LETTERS = 3


def normalize_text(value: object) -> str:
    """Normaliza um texto livre preservando acentos e caixa (importantes para exibição).

    - valores ausentes viram string vazia
    - normalização Unicode NFC
    - remove caracteres de controle e quebras de linha (\\r\\n)
    - colapsa espaços e pontuação repetida ("!!!!!" -> "!!!")
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = unicodedata.normalize("NFC", str(value))
    text = _CONTROL_CHARS.sub(" ", text)
    text = _WHITESPACE.sub(" ", text)
    text = _REPEATED_PUNCT.sub(r"\1\1\1", text)
    return text.strip()


def is_informative(text: str, min_letters: int = MIN_LETTERS) -> bool:
    """Um texto é útil para o RAG se tiver ao menos `min_letters` letras.

    Descarta comentários como ".", "10", "!!!" — sem conteúdo semântico.
    """
    return len(_LETTER.findall(text)) >= min_letters


def combine_title_message(title: str, message: str) -> str:
    """Junta título e mensagem. O título costuma ser um resumo ("Péssimo", "Recomendo")."""
    if title and message:
        if message.lower().startswith(title.lower()):
            return message
        sep = "" if title[-1] in ".!?" else "."
        return f"{title}{sep} {message}"
    return title or message


def delivery_status(row: pd.Series) -> str:
    """Classifica a entrega do pedido a partir das datas (tratando ausências)."""
    delivered = row.get("order_delivered_customer_date")
    estimated = row.get("order_estimated_delivery_date")
    status = row.get("order_status")
    if pd.isna(delivered):
        if isinstance(status, str) and status in {"canceled", "unavailable"}:
            return "cancelado"
        return "nao_entregue" if isinstance(status, str) else "desconhecido"
    if pd.isna(estimated):
        return "entregue"
    # estimated vem com hora 00:00 -> compara por dia
    return "no_prazo" if delivered.normalize() <= estimated.normalize() else "atrasado"


@dataclass
class CleaningReport:
    total_reviews: int = 0
    duplicated_review_ids_removed: int = 0
    without_text: int = 0
    non_informative_text: int = 0
    title_only: int = 0
    message_only: int = 0
    title_and_message: int = 0
    final_documents: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def clean_reviews(base: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Aplica as regras de seleção/limpeza e devolve (dataframe limpo, relatório)."""
    report = CleaningReport(total_reviews=len(base))
    df = base.copy()

    # 1) review_id duplicado: a mesma avaliação foi associada a mais de um pedido
    #    (o cliente avaliou uma única vez uma compra com vários pedidos). Mantemos
    #    uma ocorrência para não duplicar a evidência no índice.
    before = len(df)
    df = df.sort_values(["review_id", "order_purchase_timestamp"]).drop_duplicates("review_id", keep="first")
    report.duplicated_review_ids_removed = before - len(df)

    # 2) Normalização e combinação título + mensagem
    df["title_clean"] = df["review_comment_title"].map(normalize_text)
    df["message_clean"] = df["review_comment_message"].map(normalize_text)
    df["text"] = [combine_title_message(t, m) for t, m in zip(df["title_clean"], df["message_clean"])]

    has_title = df["title_clean"] != ""
    has_msg = df["message_clean"] != ""
    df["text_source"] = np.select(
        [has_title & has_msg, has_msg, has_title], ["titulo_e_mensagem", "mensagem", "titulo"], default="nenhum"
    )

    # 3) Seleção: sem texto (ausente) ou sem conteúdo informativo saem da base de conhecimento
    no_text = df["text"] == ""
    report.without_text = int(no_text.sum())
    df = df[~no_text]
    informative = df["text"].map(is_informative)
    report.non_informative_text = int((~informative).sum())
    df = df[informative].copy()

    counts = df["text_source"].value_counts()
    report.title_only = int(counts.get("titulo", 0))
    report.message_only = int(counts.get("mensagem", 0))
    report.title_and_message = int(counts.get("titulo_e_mensagem", 0))

    # 4) Metadados: tratamento de ausentes
    df["delivery_status"] = df.apply(delivery_status, axis=1)
    df["delivery_delay_days"] = (
        df["order_delivered_customer_date"].dt.normalize() - df["order_estimated_delivery_date"].dt.normalize()
    ).dt.days
    df["main_category"] = df["main_category"].fillna("sem_categoria")
    df["product_categories"] = df["product_categories"].fillna("sem_categoria")
    df["product_categories_en"] = df["product_categories_en"].fillna("unknown")
    df["customer_state"] = df["customer_state"].fillna("NA")
    df["order_status"] = df["order_status"].fillna("desconhecido")

    report.final_documents = len(df)
    return df.reset_index(drop=True), report
