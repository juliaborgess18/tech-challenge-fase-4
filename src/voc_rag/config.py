"""Configurações centralizadas do projeto, lidas de variáveis de ambiente / .env."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    return int(_env_str(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(_env_str(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
    return _env_str(name, str(default)).strip().lower() in {"1", "true", "yes", "sim"}


@dataclass(frozen=True)
class Settings:
    # Caminhos
    raw_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "raw")
    processed_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "processed")
    vectorstore_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "vectorstore")

    # Embeddings
    embedding_model: str = field(default_factory=lambda: _env_str("EMBEDDING_MODEL", "intfloat/multilingual-e5-small"))
    embedding_batch_size: int = field(default_factory=lambda: _env_int("EMBEDDING_BATCH_SIZE", 64))

    # LLM
    llm_api_key: str = field(default_factory=lambda: _env_str("LLM_API_KEY", ""))
    llm_model: str = field(default_factory=lambda: _env_str("LLM_MODEL", ""))
    llm_max_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_TOKENS", 4000))
    # Tokens de "raciocínio" do Gemini antes de responder: 0 = desligado (mais rápido), -1 = dinâmico
    llm_thinking_budget: int = field(default_factory=lambda: _env_int("LLM_THINKING_BUDGET", 0))

    # Recuperação
    top_k_candidates: int = field(default_factory=lambda: _env_int("TOP_K_CANDIDATES", 60))
    top_k_context: int = field(default_factory=lambda: _env_int("TOP_K_CONTEXT", 15))
    use_bm25: bool = field(default_factory=lambda: _env_bool("USE_BM25", True))
    use_reranker: bool = field(default_factory=lambda: _env_bool("USE_RERANKER", False))
    reranker_model: str = field(
        default_factory=lambda: _env_str("RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    )

    # Guardrails
    min_relevance: float = field(default_factory=lambda: _env_float("MIN_RELEVANCE", 0.85))
    min_evidences: int = field(default_factory=lambda: _env_int("MIN_EVIDENCES", 3))
    scope_margin: float = field(default_factory=lambda: _env_float("SCOPE_MARGIN", 0.0))
    min_term_df: int = field(default_factory=lambda: _env_int("MIN_TERM_DF", 3))
    lexical_coverage: float = field(default_factory=lambda: _env_float("LEXICAL_COVERAGE", 0.6))

    @property
    def documents_path(self) -> Path:
        return self.processed_dir / "documents.parquet"


def get_settings() -> Settings:
    return Settings()
