"""Fixtures compartilhadas: base sintética pequena e embedder falso (sem baixar modelos)."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from voc_rag.indexing.vectorstore import VectorStore
from voc_rag.retrieval.retriever import tokenize


class FakeEmbedder:
    """Bag-of-words com hashing: textos com palavras em comum ficam próximos. Determinístico."""

    model_name = "fake"
    dim = 64

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


@pytest.fixture
def documents() -> pd.DataFrame:
    rows = [
        ("r1", "Produto chegou atrasado, entrega demorou demais", 1, "atrasado", "telefonia", "2018-03"),
        ("r2", "Entrega atrasada e ninguém responde o contato", 1, "atrasado", "telefonia", "2018-04"),
        ("r3", "Atraso na entrega de mais de 10 dias", 2, "atrasado", "cama_mesa_banho", "2017-11"),
        ("r4", "Ótimo produto, entrega rápida, recomendo", 5, "no_prazo", "cama_mesa_banho", "2017-05"),
        ("r5", "Excelente qualidade, adorei a toalha", 5, "no_prazo", "cama_mesa_banho", "2018-01"),
        ("r6", "Produto veio quebrado e com defeito", 1, "no_prazo", "utilidades_domesticas", "2018-02"),
        ("r7", "Recebi o produto errado, diferente do anunciado", 2, "no_prazo", "moveis_decoracao", "2017-08"),
        ("r8", "Ótimo produto, entrega rápida, recomendo", 5, "no_prazo", "telefonia", "2018-06"),
    ]
    df = pd.DataFrame(rows, columns=["review_id", "text", "review_score", "delivery_status", "main_category", "year_month"])
    df["doc_id"] = range(len(df))
    df["order_id"] = "o_" + df["review_id"]
    df["product_categories"] = df["main_category"]
    df["customer_state"] = "SP"
    df["review_date"] = pd.to_datetime(df["year_month"] + "-15")
    return df


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def store(documents, embedder) -> VectorStore:
    return VectorStore(embedder.embed_documents(documents["text"].tolist()), documents, {"embedding_model": "fake"})
