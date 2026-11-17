"""Etapa 2 — Gera os embeddings dos documentos e persiste o índice vetorial.

Uso:
    python scripts/build_index.py            # base completa
    python scripts/build_index.py --limit 2000   # amostra rápida para testes
"""

from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

from voc_rag.config import get_settings
from voc_rag.indexing import SentenceTransformerEmbedder, VectorStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="indexa apenas os N primeiros documentos")
    args = parser.parse_args()

    settings = get_settings()
    if not settings.documents_path.exists():
        print("documents.parquet não encontrado. Rode `python scripts/build_knowledge_base.py` antes.")
        return 1

    docs = pd.read_parquet(settings.documents_path)
    if args.limit:
        docs = docs.head(args.limit)

    print(f"Carregando modelo de embeddings: {settings.embedding_model}")
    embedder = SentenceTransformerEmbedder(settings.embedding_model, batch_size=settings.embedding_batch_size)

    print(f"Gerando embeddings para {len(docs)} documentos ...")
    start = time.perf_counter()
    vectors = embedder.embed_documents(docs["text"].tolist(), show_progress=True)
    elapsed = time.perf_counter() - start

    store = VectorStore(vectors, docs, manifest={"embedding_model": settings.embedding_model})
    store.save(settings.vectorstore_dir)
    print(f"Índice salvo em {settings.vectorstore_dir} ({vectors.shape}, {elapsed:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
