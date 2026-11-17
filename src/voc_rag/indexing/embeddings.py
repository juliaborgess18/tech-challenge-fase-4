"""Criação de embeddings (modelo local, sem dependência de API).

Modelo padrão: intfloat/multilingual-e5-small
- multilíngue com bom desempenho em português (MTEB/MIRACL)
- 384 dimensões, ~118M parâmetros: roda em CPU em poucos minutos para ~42k textos curtos
- treinado para busca assimétrica pergunta -> passagem com os prefixos
  "query: " e "passage: " (obrigatórios para a família E5)
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    model_name: str

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Wrapper de sentence-transformers com prefixos E5 e vetores normalizados (cosseno = produto interno)."""

    def __init__(self, model_name: str, batch_size: int = 64, device: str | None = None):
        from sentence_transformers import SentenceTransformer  # import tardio: carregamento pesado

        self.model_name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=device)
        is_e5 = "e5" in model_name.lower()
        self._query_prefix = "query: " if is_e5 else ""
        self._passage_prefix = "passage: " if is_e5 else ""

    @property
    def dimension(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        vectors = self._model.encode(
            [self._passage_prefix + t for t in texts],
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        vector = self._model.encode(
            [self._query_prefix + text], normalize_embeddings=True, convert_to_numpy=True
        )[0]
        return vector.astype(np.float32)
