"""Etapa 1 — Constrói a base de conhecimento (data/processed/documents.parquet).

Uso:
    python scripts/build_knowledge_base.py
"""

from __future__ import annotations

import json
import sys

from voc_rag.config import get_settings
from voc_rag.ingestion import build_documents, build_review_base, clean_reviews, load_raw_tables


def main() -> int:
    settings = get_settings()
    settings.processed_dir.mkdir(parents=True, exist_ok=True)

    print(f"Lendo CSVs de {settings.raw_dir} ...")
    tables = load_raw_tables(settings.raw_dir)
    base = build_review_base(tables)

    print("Limpando e selecionando avaliações ...")
    clean_df, report = clean_reviews(base)
    docs = build_documents(clean_df)

    docs.to_parquet(settings.documents_path, index=False)
    report_path = settings.processed_dir / "cleaning_report.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    print(f"\n{len(docs)} documentos salvos em {settings.documents_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
