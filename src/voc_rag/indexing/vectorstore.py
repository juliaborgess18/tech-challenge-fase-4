"""Armazenamento e indexação vetorial.

Decisão: busca vetorial EXATA (produto interno em numpy) sobre uma matriz de
embeddings normalizados, persistida em disco junto aos documentos.

Por quê, em vez de um banco vetorial (Chroma, FAISS-HNSW, pgvector)?
- ~42k vetores x 384 dims = ~64 MB: uma consulta exata leva poucos milissegundos em CPU
- busca exata não tem perda de recall de índices aproximados (ANN)
- filtros por metadados são aplicados ANTES da busca (pre-filtering) com uma
  máscara booleana do pandas, o que é exato e simples de explicar
- zero infraestrutura: o avaliador reproduz tudo com `pip install`

Para escalar (milhões de avaliações), a interface `search(query_vec, k, mask)`
pode ser reimplementada sobre FAISS/Qdrant/pgvector sem mudar o resto do pipeline.

Arquivos gerados em data/vectorstore/:
    embeddings.npy       matriz [n_docs, dim] salva em float16 (carregada como float32), linhas alinhadas a documents.parquet
    documents.parquet    documentos + metadados
    manifest.json        modelo de embedding, dimensão, contagem, data de criação
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


class VectorStore:
    def __init__(self, embeddings: np.ndarray, documents: pd.DataFrame, manifest: dict | None = None):
        if len(embeddings) != len(documents):
            raise ValueError(f"embeddings ({len(embeddings)}) e documentos ({len(documents)}) desalinhados")
        self.embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.documents = documents.reset_index(drop=True)
        self.manifest = manifest or {}

    def __len__(self) -> int:
        return len(self.documents)

    # ------------------------------------------------------------------ busca
    def search(self, query_vector: np.ndarray, k: int, mask: np.ndarray | None = None) -> list[tuple[int, float]]:
        """Retorna [(posição do documento, similaridade cosseno)] em ordem decrescente.

        `mask` (bool, tamanho n_docs) restringe a busca aos documentos permitidos pelos filtros.
        """
        scores = self.embeddings @ query_vector.astype(np.float32)
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
            k = min(k, int(mask.sum()))
        k = min(k, len(scores))
        if k <= 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top]

    # ----------------------------------------------------------- persistência
    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        # float16 em disco: metade do tamanho (~32 MB), cabe no GitHub para o deploy.
        # Ao carregar, volta para float32; a perda de precisão não altera a busca (ver avaliação).
        np.save(directory / "embeddings.npy", self.embeddings.astype(np.float16))
        self.documents.to_parquet(directory / "documents.parquet", index=False)
        manifest = {
            **self.manifest,
            "n_documents": len(self),
            "dimension": int(self.embeddings.shape[1]),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        self.manifest = manifest

    @classmethod
    def load(cls, directory: Path, expected_model: str | None = None) -> "VectorStore":
        directory = Path(directory)
        if not (directory / "embeddings.npy").exists():
            raise FileNotFoundError(
                f"Índice não encontrado em {directory}. Rode `python scripts/build_index.py` primeiro."
            )
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if expected_model and manifest.get("embedding_model") != expected_model:
            raise ValueError(
                f"Índice criado com '{manifest.get('embedding_model')}', mas a configuração usa "
                f"'{expected_model}'. Recrie o índice com `python scripts/build_index.py`."
            )
        embeddings = np.load(directory / "embeddings.npy")
        documents = pd.read_parquet(directory / "documents.parquet")
        return cls(embeddings, documents, manifest)
