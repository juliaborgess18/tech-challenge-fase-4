"""Filtros por metadados (desafio adicional).

Os filtros são aplicados ANTES da busca vetorial (pre-filtering). Isso garante que,
por exemplo, "problemas em avaliações com nota baixa" recupere apenas avaliações
nota 1-2, em vez de depender de o embedding "entender" a nota — informação que
NÃO está no texto do comentário.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd


@dataclass
class MetadataFilter:
    min_score: int | None = None
    max_score: int | None = None
    categories: list[str] = field(default_factory=list)  # nomes em português (ex.: "cama_mesa_banho")
    delivery_status: list[str] = field(default_factory=list)  # no_prazo | atrasado | nao_entregue | cancelado
    period_start: str | None = None  # "YYYY-MM"
    period_end: str | None = None  # "YYYY-MM"
    customer_states: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not any(
            [
                self.min_score,
                self.max_score,
                self.categories,
                self.delivery_status,
                self.period_start,
                self.period_end,
                self.customer_states,
            ]
        )

    def merge(self, override: "MetadataFilter | None") -> "MetadataFilter":
        """Campos preenchidos em `override` (ex.: escolhidos na interface) têm prioridade."""
        if override is None:
            return self
        merged = asdict(self)
        for key, value in asdict(override).items():
            if value not in (None, [], ""):
                merged[key] = value
        return MetadataFilter(**merged)

    def describe(self) -> str:
        parts = []
        if self.min_score or self.max_score:
            parts.append(f"nota {self.min_score or 1}-{self.max_score or 5}")
        if self.categories:
            parts.append("categorias: " + ", ".join(self.categories))
        if self.delivery_status:
            parts.append("entrega: " + ", ".join(self.delivery_status))
        if self.period_start or self.period_end:
            parts.append(f"período: {self.period_start or 'início'} a {self.period_end or 'fim'}")
        if self.customer_states:
            parts.append("UF: " + ", ".join(self.customer_states))
        return "; ".join(parts) if parts else "nenhum"

    def to_dict(self) -> dict:
        return asdict(self)


def apply_filter(documents: pd.DataFrame, flt: MetadataFilter | None) -> np.ndarray | None:
    """Máscara booleana dos documentos que passam no filtro (None = sem filtro)."""
    if flt is None or flt.is_empty():
        return None
    mask = pd.Series(True, index=documents.index)
    if flt.min_score is not None:
        mask &= documents["review_score"] >= flt.min_score
    if flt.max_score is not None:
        mask &= documents["review_score"] <= flt.max_score
    if flt.categories:
        # um pedido pode ter várias categorias ("a|b"): basta uma coincidir
        cats = documents["product_categories"].str.split("|")
        wanted = set(flt.categories)
        mask &= cats.map(lambda cs: bool(wanted.intersection(cs)))
    if flt.delivery_status:
        mask &= documents["delivery_status"].isin(flt.delivery_status)
    if flt.period_start:
        mask &= documents["year_month"] >= flt.period_start
    if flt.period_end:
        mask &= documents["year_month"] <= flt.period_end
    if flt.customer_states:
        mask &= documents["customer_state"].isin(flt.customer_states)
    return mask.to_numpy()
