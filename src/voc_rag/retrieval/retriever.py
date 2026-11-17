"""Recuperação semântica (+ lexical opcional) das avaliações.

Estratégia: busca HÍBRIDA com Reciprocal Rank Fusion (RRF).
- Densa (embeddings E5): captura paráfrases ("demorou demais" ~ "atraso na entrega").
- Lexical (BM25): reforça termos exatos e curtos, comuns em reviews ("não recebi",
  "veio quebrado", nomes de produtos), onde embeddings de frases curtas oscilam.
- RRF combina os rankings sem precisar calibrar escalas diferentes de score:
      rrf(d) = Σ 1 / (k + rank_i(d)),  k = 60
A similaridade densa (cosseno) é preservada em cada candidato porque é usada
pelos guardrails para decidir se há evidência suficiente.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

from voc_rag.indexing.embeddings import Embedder
from voc_rag.indexing.vectorstore import VectorStore
from voc_rag.retrieval.filters import apply_filter
from voc_rag.retrieval.query_processing import ProcessedQuery, strip_accents

RRF_K = 60

STOPWORDS_PT = set(
    """a o as os um uma uns umas de do da dos das em no na nos nas por pelo pela pelos pelas para pra com sem
    e ou que se ja nao sim mas mais muito muita muitos muitas foi era ser ter tem tinha estou esta estao
    eu me meu minha voce ele ela eles elas isso isto esse essa este esta aquele aquela lhe ao aos
    qual quais quem como quando onde porque sobre entre ate apos tambem so seu sua seus suas nos
    sao clientes cliente avaliacoes avaliacao relatam relatados relatado principais principal""".split()
)


def tokenize(text: str) -> list[str]:
    text = strip_accents(text.lower())
    return [t for t in re.findall(r"[a-z0-9]+", text) if len(t) > 1 and t not in STOPWORDS_PT]


STEM_LEN = 5


def stem(token: str) -> str:
    """Stemming por prefixo: agrupa flexões ("atraso", "atrasou", "atrasada" -> "atras")."""
    return token[:STEM_LEN]


@dataclass
class Candidate:
    position: int  # posição no VectorStore
    dense_score: float  # similaridade cosseno com a pergunta (0-1)
    bm25_score: float = 0.0
    fused_score: float = 0.0
    rerank_score: float | None = None
    dense_rank: int | None = None
    bm25_rank: int | None = None


class HybridRetriever:
    def __init__(self, store: VectorStore, embedder: Embedder, use_bm25: bool = True):
        self.store = store
        self.embedder = embedder
        self.use_bm25 = use_bm25
        self._bm25 = None
        corpus = [tokenize(t) for t in store.documents["text"].tolist()]
        # estatísticas lexicais usadas pelos guardrails
        self.doc_stems: list[frozenset[str]] = [frozenset(stem(t) for t in toks) for toks in corpus]
        self.stem_df: Counter = Counter(s for stems in self.doc_stems for s in stems)
        if use_bm25:
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi(corpus)

    def term_document_frequency(self, term: str) -> int:
        return self.stem_df.get(stem(term), 0)

    def lexical_support(self, key_terms: list[str], mask: np.ndarray | None = None, coverage: float = 0.6) -> int:
        """Nº de documentos (dentro do filtro) que contêm ao menos `coverage` dos termos-chave."""
        stems = {stem(t) for t in key_terms}
        if not stems:
            return -1  # sem termos-chave: verificação não se aplica
        needed = max(1, math.ceil(coverage * len(stems)))
        allowed = range(len(self.doc_stems)) if mask is None else np.flatnonzero(mask)
        return sum(1 for i in allowed if len(stems & self.doc_stems[i]) >= needed)

    @property
    def known_categories(self) -> set[str]:
        cats = self.store.documents["product_categories"].str.split("|").explode()
        return set(cats.dropna().unique())

    def retrieve(self, query: ProcessedQuery, k: int = 60) -> tuple[list[Candidate], dict]:
        """Retorna (candidatos ordenados, estatísticas para o trace)."""
        mask = apply_filter(self.store.documents, query.filters)
        n_allowed = len(self.store) if mask is None else int(mask.sum())
        stats = {"documents_after_filters": n_allowed, "documents_total": len(self.store)}
        if n_allowed == 0:
            return [], stats

        query_vector = self.embedder.embed_query(query.embedding_text)
        all_dense = self.store.embeddings @ query_vector

        dense_hits = self.store.search(query_vector, k=k, mask=mask)
        candidates: dict[int, Candidate] = {}
        for rank, (pos, score) in enumerate(dense_hits, start=1):
            candidates[pos] = Candidate(position=pos, dense_score=score, dense_rank=rank)

        if self._bm25 is not None:
            tokens = tokenize(query.bm25_text)
            if tokens:
                bm25_scores = self._bm25.get_scores(tokens)
                if mask is not None:
                    bm25_scores = np.where(mask, bm25_scores, -np.inf)
                kk = min(k, n_allowed)
                top = np.argpartition(-bm25_scores, kk - 1)[:kk]
                top = top[np.argsort(-bm25_scores[top])]
                for rank, pos in enumerate(top, start=1):
                    pos = int(pos)
                    if bm25_scores[pos] <= 0:
                        break
                    cand = candidates.get(pos) or Candidate(position=pos, dense_score=float(all_dense[pos]))
                    cand.bm25_score = float(bm25_scores[pos])
                    cand.bm25_rank = rank
                    candidates[pos] = cand

        for cand in candidates.values():
            fused = 0.0
            if cand.dense_rank:
                fused += 1.0 / (RRF_K + cand.dense_rank)
            if cand.bm25_rank:
                fused += 1.0 / (RRF_K + cand.bm25_rank)
            cand.fused_score = fused

        ranked = sorted(candidates.values(), key=lambda c: c.fused_score, reverse=True)
        stats["dense_candidates"] = len(dense_hits)
        stats["bm25_candidates"] = sum(1 for c in ranked if c.bm25_rank)
        stats["fused_candidates"] = len(ranked)
        stats["best_dense_score"] = round(max((c.dense_score for c in ranked), default=0.0), 4)
        return ranked, stats
