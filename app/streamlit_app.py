"""Interface de demonstração — Voice of Customer Intelligence.

Executar:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import hashlib
import math
import os
import threading
import time

import pandas as pd
import streamlit as st

from voc_rag.config import get_settings
from voc_rag.generation import LLMClient
from voc_rag.indexing import SentenceTransformerEmbedder, VectorStore
from voc_rag.pipeline import VoCPipeline
from voc_rag.retrieval import CrossEncoderReranker, MetadataFilter

st.set_page_config(page_title="Voice of Customer RAG", page_icon="💬", layout="wide")


def secrets_to_env() -> None:
    """No Streamlit Cloud a configuração vem dos Secrets (não há .env): copia para o ambiente,
    onde o config.py lê. Aceita chaves na raiz ou dentro de uma seção ([gemini], [llm]...) e nomes
    em minúsculas. Localmente o .env tem prioridade e, sem secrets.toml, nada acontece."""
    try:
        items = list(st.secrets.items())
    except Exception:  # sem secrets.toml (execução local)
        return
    flat: list[tuple[str, object]] = []
    for key, value in items:
        if hasattr(value, "items"):  # seção do TOML
            flat.extend(value.items())
        else:
            flat.append((key, value))
    for key, value in flat:
        name = str(key).strip().upper()
        if isinstance(value, (str, int, float, bool)) and not os.environ.get(name):
            os.environ[name] = str(value).strip()


secrets_to_env()

EXAMPLES = {
    "Investigação de reclamações": "Quais são os principais problemas relatados pelos clientes em suas avaliações?",
    "Tema específico": "O que os clientes relatam sobre problemas relacionados à entrega?",
    "Experiências positivas": "Quais aspectos da experiência de compra são mais elogiados pelos clientes?",
    "Insatisfação": "Quais padrões podem ser identificados nas avaliações de clientes insatisfeitos?",
    "Exploração por evidências": "Quais avaliações sustentam a conclusão de que existem problemas recorrentes relacionados ao prazo de entrega?",
    "Qualidade dos produtos": "Existem padrões recorrentes nas avaliações relacionadas à qualidade dos produtos?",
}

STATUS_UI = {
    "answered": ("success", "Resposta fundamentada nas avaliações"),
    "insufficient_evidence": ("warning", "Evidências insuficientes"),
    "out_of_scope": ("error", "Pergunta fora do escopo da base"),
    "retrieval_only": ("info", "Somente recuperação (LLM não configurado ou indisponível)"),
}

# Progresso (%) e mensagem exibidos no início de cada etapa do pipeline.
ASK_STEPS = {
    "query_processing": (5, "Interpretando a pergunta..."),
    "retrieval": (15, "Buscando avaliações relevantes..."),
    "rerank": (30, "Reordenando as avaliações..."),
    "guardrail": (35, "Verificando se há evidências suficientes..."),
    "selection": (45, "Selecionando as melhores evidências..."),
    "generation": (55, "Gerando resposta com o Gemini..."),
}

DELIVERY_OPTIONS = {
    "no_prazo": "Entregue no prazo",
    "atrasado": "Entregue com atraso",
    "nao_entregue": "Não entregue",
    "cancelado": "Cancelado",
}


# Carregamento em etapas (cada uma em cache), para mostrar o progresso ao usuário.
@st.cache_resource(show_spinner=False)
def load_store() -> VectorStore:
    s = get_settings()
    return VectorStore.load(s.vectorstore_dir, expected_model=s.embedding_model)


@st.cache_resource(show_spinner=False)
def load_embedder() -> SentenceTransformerEmbedder:
    s = get_settings()
    return SentenceTransformerEmbedder(s.embedding_model, batch_size=s.embedding_batch_size)


@st.cache_resource(show_spinner=False)
def build_pipeline(_store: VectorStore, _embedder: SentenceTransformerEmbedder, llm_fingerprint: str) -> VoCPipeline:
    # llm_fingerprint só entra na chave do cache: se a chave/modelo mudarem nos Secrets, o pipeline é refeito
    s = get_settings()
    llm = LLMClient(s.llm_api_key, s.llm_model, s.llm_max_tokens, s.llm_thinking_budget)
    reranker = CrossEncoderReranker(s.reranker_model) if s.use_reranker else None
    return VoCPipeline(_store, _embedder, llm, s, reranker)


def load_pipeline_with_progress() -> VoCPipeline:
    bar = st.progress(0, text="Carregando o índice de avaliações... (etapa 1 de 3)")
    store = load_store()
    bar.progress(33, text="Carregando o modelo de embeddings... (etapa 2 de 3)")
    embedder = load_embedder()
    bar.progress(66, text="Preparando a busca e a conexão com o Gemini... (etapa 3 de 3)")
    s = get_settings()
    fingerprint = hashlib.sha256(f"{s.llm_api_key}|{s.llm_model}|{s.llm_thinking_budget}".encode()).hexdigest()[:12]
    pipeline = build_pipeline(store, embedder, fingerprint)
    bar.progress(100, text="Pronto!")
    bar.empty()
    return pipeline


def ask_with_progress(pipeline: VoCPipeline, question: str, **kwargs):
    """Roda o pipeline em uma thread e atualiza uma barra de progresso enquanto espera."""
    state: dict = {"step": None, "resp": None, "error": None}

    def worker():
        try:
            state["resp"] = pipeline.ask(question, on_step=lambda s: state.update(step=s), **kwargs)
        except Exception as exc:  # repassado para a thread do Streamlit
            state["error"] = exc

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    bar = st.progress(0, text="Iniciando...")
    gen_start = None
    while thread.is_alive():
        step = state["step"]
        pct, text = ASK_STEPS.get(step, (0, "Iniciando..."))
        if step == "generation":
            # duração da geração é imprevisível: avança devagar, sem chegar a 100%
            gen_start = gen_start or time.monotonic()
            elapsed = time.monotonic() - gen_start
            pct += int(40 * (1 - math.exp(-elapsed / 4)))
            text = f"Gerando resposta com o Gemini... {elapsed:.0f} s (costuma levar poucos segundos)"
        bar.progress(pct, text=text)
        time.sleep(0.25)
    bar.empty()

    if state["error"]:
        raise state["error"]
    return state["resp"]


# ------------------------------------------------------------------ main
st.title("Voice of Customer Intelligence")
st.caption("Perguntas em linguagem natural sobre ~42 mil avaliações de clientes do Olist, respondidas com evidências.")

cols = st.columns(3)
for i, (label, q) in enumerate(EXAMPLES.items()):
    if cols[i % 3].button(label, width="stretch"):
        st.session_state["question"] = q

question = st.text_area("Pergunta", key="question", height=80, placeholder="Ex.: O que os clientes reclamam sobre entrega?")

# Enquanto os modelos carregam, o botão aparece desativado e a barra mostra a etapa atual.
ask_slot = st.empty()
ask_slot.button("⏳ Carregando modelos...", disabled=True, key="ask_loading")

try:
    pipeline = load_pipeline_with_progress()
except FileNotFoundError as exc:
    ask_slot.empty()
    st.error(str(exc))
    st.code("python scripts/build_knowledge_base.py\npython scripts/build_index.py")
    st.stop()

run = ask_slot.button("Perguntar", type="primary", disabled=not question.strip(), key="ask")

docs = pipeline.store.documents

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Modo de resposta")
    llm_ready = pipeline.llm.is_configured
    use_llm = st.toggle(
        "Gerar resposta com o Gemini",
        value=llm_ready,
        disabled=not llm_ready,
        key="use_llm",
        help=(
            "Ligado: o Gemini escreve a resposta citando as avaliações [E1], [E2]…\n\n"
            "Desligado: só a recuperação — mostra as avaliações encontradas e o trace, sem chamar o LLM "
            "(não gasta cota)."
            if llm_ready
            else "Configure LLM_API_KEY e LLM_MODEL (no .env ou nos Secrets) para ativar."
        ),
    )
    use_llm = use_llm and llm_ready

    st.header("Filtros de metadados")
    auto_filters = st.toggle(
        "Extrair filtros da pergunta",
        value=True,
        help="Ex.: 'nota baixa' → notas 1-2; 'celulares' → telefonia; '2018' → período.",
    )
    score_range = st.slider("Nota da avaliação", 1, 5, (1, 5))
    categories = st.multiselect("Categoria do produto", sorted(pipeline.known_categories))
    delivery = st.multiselect("Situação da entrega", list(DELIVERY_OPTIONS), format_func=DELIVERY_OPTIONS.get)
    months = sorted(docs["year_month"].unique())
    period = st.select_slider("Período da compra", options=months, value=(months[0], months[-1]))
    top_k = st.slider("Evidências no contexto", 5, 30, pipeline.settings.top_k_context)

    st.divider()
    missing = [n for n, v in (("LLM_API_KEY", pipeline.llm.api_key), ("LLM_MODEL", pipeline.llm.model)) if not v]
    llm_label = "`" + pipeline.llm.model + "`" if not missing else "não configurado (faltando: " + ", ".join(missing) + ")"
    st.caption(
        f"Base: **{len(docs):,}** avaliações".replace(",", ".") + f" com texto · Embeddings: `{pipeline.settings.embedding_model}` · "
        f"LLM: {llm_label}"
    )

manual_filter = MetadataFilter(
    min_score=score_range[0] if score_range != (1, 5) else None,
    max_score=score_range[1] if score_range != (1, 5) else None,
    categories=categories,
    delivery_status=delivery,
    period_start=period[0] if period[0] != months[0] else None,
    period_end=period[1] if period[1] != months[-1] else None,
)

if run and question.strip():
    resp = ask_with_progress(
        pipeline, question, filters=manual_filter, auto_filters=auto_filters, top_k=top_k, generate=use_llm
    )

    kind, label = STATUS_UI[resp.status.value]
    if resp.status.value == "retrieval_only" and llm_ready and not use_llm:
        # o usuário desligou o Gemini de propósito
        st.info("Somente recuperação (Gemini desligado)")
        st.markdown(
            "Abaixo estão as avaliações que a busca encontrou — exatamente o contexto que seria enviado ao Gemini. "
            "Ligue **Gerar resposta com o Gemini** na barra lateral para ver a resposta escrita com as citações."
        )
    else:
        getattr(st, kind)(label)
        st.markdown(resp.answer)
    for w in resp.warnings:
        st.warning(w)

    st.caption(f"Filtros aplicados: {resp.trace.filters_applied}")

    if resp.evidences:
        st.subheader(f"Evidências recuperadas ({len(resp.evidences)})")
        n_cited = len(resp.cited_evidences)
        if n_cited:
            st.caption(f"{n_cited} evidência(s) citadas na resposta estão marcadas com ✅")

        table = pd.DataFrame(
            [
                {
                    "": "✅" if e.cited else "",
                    "id": e.evidence_id,
                    "nota": "★" * e.review_score,
                    "avaliação": e.text,
                    "categoria": e.main_category,
                    "entrega": DELIVERY_OPTIONS.get(e.delivery_status, e.delivery_status),
                    "período": e.year_month,
                    "similaridade": e.similarity,
                    "idênticas": e.similar_reviews,
                    "review_id": e.review_id,
                }
                for e in resp.evidences
            ]
        )
        st.dataframe(
            table,
            hide_index=True,
            width="stretch",
            column_config={
                "avaliação": st.column_config.TextColumn(width="large"),
                "similaridade": st.column_config.ProgressColumn(min_value=0.7, max_value=1.0, format="%.3f"),
            },
        )

        c1, c2 = st.columns(2)
        c1.bar_chart(table["nota"].str.len().value_counts().sort_index(), x_label="nota", y_label="evidências")
        c2.bar_chart(table["entrega"].value_counts(), horizontal=True)

    with st.expander("Como a resposta foi construída (trace do pipeline)"):
        t = resp.trace
        st.markdown("**1. Transformação da pergunta**")
        st.write(t.processed_query.get("notes") or "nenhuma transformação aplicada")
        st.markdown("**2. Recuperação (densa + BM25 + RRF)**")
        st.json(t.retrieval_stats)
        st.markdown("**3. Guardrail de evidência**")
        st.json(t.guardrail)
        st.markdown("**4. Seleção**")
        st.json(t.selection_stats)
        st.markdown("**Tempos (ms)**")
        st.json(t.timings_ms)
        if t.context_preview:
            st.markdown("**5. Contexto enviado ao modelo**")
            st.code(t.context_preview, language="text")
