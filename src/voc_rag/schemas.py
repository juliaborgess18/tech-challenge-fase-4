"""Estruturas de saída do pipeline (resposta, evidências e trace de execução)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum


class AnswerStatus(str, Enum):
    ANSWERED = "answered"  # resposta gerada com base em evidências
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"  # há algo relacionado, mas não o bastante
    OUT_OF_SCOPE = "out_of_scope"  # a pergunta não é sobre a base de avaliações
    RETRIEVAL_ONLY = "retrieval_only"  # LLM não configurado: devolve só as evidências


@dataclass
class Evidence:
    evidence_id: str  # rótulo usado nas citações da resposta ("E1", "E2"...)
    review_id: str
    order_id: str
    text: str
    review_score: int
    review_date: str
    year_month: str
    main_category: str
    delivery_status: str
    customer_state: str
    similarity: float
    fused_score: float
    rerank_score: float | None = None
    similar_reviews: int = 0  # nº de avaliações idênticas colapsadas nesta evidência
    cited: bool = False  # se a resposta final citou esta evidência


@dataclass
class RetrievalTrace:
    """Registro de cada etapa, para demonstrar o pipeline e depurar."""

    original_question: str = ""
    processed_query: dict = field(default_factory=dict)
    filters_applied: str = "nenhum"
    retrieval_stats: dict = field(default_factory=dict)
    selection_stats: dict = field(default_factory=dict)
    guardrail: dict = field(default_factory=dict)
    context_preview: str = ""
    timings_ms: dict = field(default_factory=dict)
    llm_model: str | None = None


@dataclass
class RAGResponse:
    question: str
    answer: str
    status: AnswerStatus
    evidences: list[Evidence] = field(default_factory=list)
    trace: RetrievalTrace = field(default_factory=RetrievalTrace)
    warnings: list[str] = field(default_factory=list)

    @property
    def cited_evidences(self) -> list[Evidence]:
        return [e for e in self.evidences if e.cited]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["status"] = self.status.value
        return data
