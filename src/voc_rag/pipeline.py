"""Orquestração do pipeline RAG.

    pergunta
      -> process_query        (normalização, filtros de metadados, expansão)
      -> HybridRetriever      (embedding da pergunta + busca densa + BM25 + RRF)
      -> [reranker opcional]
      -> assess_evidence      (guardrail pré-geração: escopo, termos desconhecidos, suporte lexical e semântico)
      -> select_evidence      (relevância mínima, deduplicação, top-k)
      -> build_context        (evidências rotuladas [E1..En] + resumo da amostra)
      -> LLMClient.generate   (resposta fundamentada com citações)
      -> validate_answer      (guardrail pós-geração: citações válidas / recusa)
      -> RAGResponse          (resposta + evidências + trace de cada etapa)
"""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import contextmanager

from voc_rag.config import Settings, get_settings
from voc_rag.generation import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    EvidenceSignals,
    LLMClient,
    LLMNotConfiguredError,
    ScopeClassifier,
    assess_evidence,
    count_relevant,
    build_context,
    to_evidences,
    validate_answer,
)
from voc_rag.generation.guardrails import MESSAGES
from voc_rag.indexing import Embedder, SentenceTransformerEmbedder, VectorStore
from voc_rag.retrieval import (
    CrossEncoderReranker,
    HybridRetriever,
    MetadataFilter,
    apply_filter,
    process_query,
    select_evidence,
)
from voc_rag.schemas import AnswerStatus, RAGResponse, RetrievalTrace


class _Timer:
    def __init__(self, on_step: Callable[[str], None] | None = None) -> None:
        self.timings: dict[str, float] = {}
        self.on_step = on_step

    @contextmanager
    def step(self, name: str):
        if self.on_step:
            self.on_step(name)
        start = time.perf_counter()
        yield
        self.timings[name] = round((time.perf_counter() - start) * 1000, 1)


class VoCPipeline:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        llm: LLMClient,
        settings: Settings,
        reranker: CrossEncoderReranker | None = None,
        scope_classifier: ScopeClassifier | None = None,
    ):
        self.settings = settings
        self.store = store
        self.llm = llm
        self.reranker = reranker
        self.retriever = HybridRetriever(store, embedder, use_bm25=settings.use_bm25)
        self.scope = scope_classifier or ScopeClassifier(embedder)
        self.known_categories = self.retriever.known_categories

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "VoCPipeline":
        settings = settings or get_settings()
        store = VectorStore.load(settings.vectorstore_dir, expected_model=settings.embedding_model)
        embedder = SentenceTransformerEmbedder(settings.embedding_model, batch_size=settings.embedding_batch_size)
        llm = LLMClient(settings.llm_api_key, settings.llm_model, settings.llm_max_tokens, settings.llm_thinking_budget)
        reranker = CrossEncoderReranker(settings.reranker_model) if settings.use_reranker else None
        return cls(store, embedder, llm, settings, reranker)

    def ask(
        self,
        question: str,
        filters: MetadataFilter | None = None,
        auto_filters: bool = True,
        generate: bool = True,
        top_k: int | None = None,
        on_step: Callable[[str], None] | None = None,
    ) -> RAGResponse:
        """Responde a uma pergunta.

        filters:      filtros explícitos (ex.: da interface); têm prioridade sobre os automáticos
        auto_filters: extrai filtros da própria pergunta ("nota baixa", "2018", "celulares"...)
        generate:     False = só recuperação (útil para avaliar o retriever sem gastar tokens)
        on_step:      callback chamado no início de cada etapa (ex.: barra de progresso na interface)
        """
        s = self.settings
        top_k = top_k or s.top_k_context
        timer = _Timer(on_step)
        trace = RetrievalTrace(original_question=question)

        # 1. Transformação da pergunta
        with timer.step("query_processing"):
            pq = process_query(question, self.known_categories, auto_filters=auto_filters)
            pq.filters = pq.filters.merge(filters)
        trace.processed_query = pq.to_dict()
        trace.filters_applied = pq.filters.describe()

        # 2. Recuperação semântica + lexical
        with timer.step("retrieval"):
            candidates, retrieval_stats = self.retriever.retrieve(pq, k=s.top_k_candidates)
        trace.retrieval_stats = retrieval_stats

        # 3. Reranking (opcional)
        if self.reranker and candidates:
            with timer.step("rerank"):
                texts = [self.store.documents.iloc[c.position]["text"] for c in candidates]
                candidates = self.reranker.rerank(question, candidates, texts)

        # 4. Guardrail pré-geração
        with timer.step("guardrail"):
            mask = apply_filter(self.store.documents, pq.filters)
            signals = EvidenceSignals(
                n_candidates=len(candidates),
                scope_margin=round(self.scope.margin(question), 4),
                key_terms=pq.key_terms,
                unknown_terms=[
                    t for t in pq.key_terms if self.retriever.term_document_frequency(t) < s.min_term_df
                ],
                lexical_support=self.retriever.lexical_support(pq.key_terms, mask, s.lexical_coverage),
                relevant_count=count_relevant(candidates, s.min_relevance),
                best_similarity=round(max((c.dense_score for c in candidates), default=0.0), 4),
                filters_active=not pq.filters.is_empty(),
                slice_summary=not pq.key_terms and not pq.filters.is_empty(),
            )
            decision = assess_evidence(
                signals, min_evidences=s.min_evidences, scope_margin=s.scope_margin, min_relevance=s.min_relevance
            )
        trace.guardrail = decision.to_dict()

        # 5. Seleção das evidências mais relevantes
        with timer.step("selection"):
            texts_by_pos = {c.position: self.store.documents.iloc[c.position]["text"] for c in candidates}
            # recorte sem tema: o filtro já define a relevância, então não aplicamos o corte semântico
            min_rel = 0.0 if signals.slice_summary else s.min_relevance
            selected, duplicates, selection_stats = select_evidence(
                candidates, texts_by_pos, top_k=top_k, min_relevance=min_rel, embeddings=self.store.embeddings
            )
            evidences = to_evidences(selected, self.store.documents, duplicates)
        trace.selection_stats = selection_stats

        if not decision.allowed:
            trace.timings_ms = timer.timings
            # mostra o que foi encontrado (se algo) para transparência, mas não gera resposta
            return RAGResponse(question, MESSAGES[decision.status], decision.status, evidences, trace)

        # 6. Construção do contexto
        evidence_block, sample_summary = build_context(evidences)
        user_prompt = USER_PROMPT_TEMPLATE.format(
            question=question,
            filters=trace.filters_applied,
            sample_summary=sample_summary,
            evidences=evidence_block,
        )
        trace.context_preview = user_prompt

        # 7. Geração
        if not generate or not self.llm.is_configured:
            trace.timings_ms = timer.timings
            msg = (
                "LLM não configurado — exibindo apenas as avaliações recuperadas. "
                "Configure LLM_API_KEY e LLM_MODEL no arquivo .env (local) ou nos Secrets do Streamlit Cloud."
                if generate
                else "Modo somente recuperação."
            )
            return RAGResponse(question, msg, AnswerStatus.RETRIEVAL_ONLY, evidences, trace)

        warnings: list[str] = []
        try:
            with timer.step("generation"):
                raw_answer = self.llm.generate(SYSTEM_PROMPT, user_prompt)
        except LLMNotConfiguredError as exc:
            trace.timings_ms = timer.timings
            return RAGResponse(question, str(exc), AnswerStatus.RETRIEVAL_ONLY, evidences, trace)
        trace.llm_model = self.llm.model

        # 8. Guardrail pós-geração
        answer, status, cited_ids, post_warnings = validate_answer(raw_answer, {e.evidence_id for e in evidences})
        warnings.extend(post_warnings)
        for ev in evidences:
            ev.cited = ev.evidence_id in cited_ids

        trace.timings_ms = timer.timings
        return RAGResponse(question, answer, status, evidences, trace, warnings)
