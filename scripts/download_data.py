"""Baixa o Brazilian E-Commerce Public Dataset by Olist para data/raw/.

Opção A: Kaggle API (requer `pip install kaggle` e o token em ~/.kaggle/kaggle.json):
    python scripts/download_data.py

Opção B (manual):
    1. Baixe o zip em https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
    2. Extraia os CSVs em data/raw/
    (ou use: python scripts/download_data.py --from-dir CAMINHO/DA/PASTA/EXTRAIDA)
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from voc_rag.config import get_settings
from voc_rag.ingestion.loader import FILES

DATASET = "olistbr/brazilian-ecommerce"


def copy_from_dir(source: Path, target: Path) -> None:
    for name in FILES.values():
        src = source / name
        if not src.exists():
            raise FileNotFoundError(f"{src} não encontrado")
        shutil.copy2(src, target / name)
        print(f"copiado: {name}")


def download_from_kaggle(target: Path) -> None:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise SystemExit("Instale a Kaggle API: pip install kaggle (ou use --from-dir)") from exc
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(DATASET, path=str(target), unzip=True, quiet=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from-dir", type=Path, help="copia os CSVs de uma pasta local já extraída")
    args = parser.parse_args()

    target = get_settings().raw_dir
    target.mkdir(parents=True, exist_ok=True)
    if args.from_dir:
        copy_from_dir(args.from_dir, target)
    else:
        download_from_kaggle(target)

    missing = [f for f in FILES.values() if not (target / f).exists()]
    if missing:
        print(f"Atenção, arquivos ausentes: {missing}")
        return 1
    print(f"Dataset pronto em {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
