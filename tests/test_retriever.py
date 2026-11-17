import numpy as np

from voc_rag.retrieval.filters import MetadataFilter
from voc_rag.retrieval.query_processing import process_query
from voc_rag.retrieval.reranker import select_evidence
from voc_rag.retrieval.retriever import HybridRetriever, tokenize

CATS = {"telefonia", "cama_mesa_banho", "utilidades_domesticas", "moveis_decoracao"}


def test_tokenize_removes_accents_and_stopwords():
    assert tokenize("Não recebi o produto até agora!") == ["recebi", "produto", "agora"]


def test_vectorstore_search_respects_mask(store):
    q = np.ones(store.embeddings.shape[1], dtype=np.float32)
    mask = np.zeros(len(store), dtype=bool)
    mask[[1, 3]] = True
    hits = store.search(q / np.linalg.norm(q), k=5, mask=mask)
    assert {pos for pos, _ in hits} == {1, 3}
    assert hits[0][1] >= hits[1][1]


def test_vectorstore_save_load(store, tmp_path):
    store.save(tmp_path)
    loaded = type(store).load(tmp_path, expected_model="fake")
    assert len(loaded) == len(store)
    # salvo em float16 (metade do tamanho): volta como float32, com arredondamento pequeno
    assert loaded.embeddings.dtype == np.float32
    assert np.allclose(loaded.embeddings, store.embeddings, atol=1e-3)


def test_hybrid_retrieval_finds_delivery_reviews(store, embedder):
    retriever = HybridRetriever(store, embedder, use_bm25=True)
    pq = process_query("entrega atrasada", CATS, auto_filters=False)
    candidates, stats = retriever.retrieve(pq, k=3)
    top_ids = [store.documents.iloc[c.position]["review_id"] for c in candidates[:3]]
    assert set(top_ids) <= {"r1", "r2", "r3"}
    assert stats["documents_after_filters"] == len(store)


def test_retrieval_with_filter_returns_only_allowed(store, embedder):
    retriever = HybridRetriever(store, embedder, use_bm25=True)
    pq = process_query("produto", CATS, auto_filters=False)
    pq.filters = MetadataFilter(min_score=5)
    candidates, _ = retriever.retrieve(pq, k=10)
    assert {store.documents.iloc[c.position]["review_score"] for c in candidates} == {5}


def test_select_evidence_collapses_duplicates(store, embedder):
    retriever = HybridRetriever(store, embedder, use_bm25=False)
    pq = process_query("ótimo produto entrega rápida recomendo", CATS, auto_filters=False)
    candidates, _ = retriever.retrieve(pq, k=8)
    texts = {c.position: store.documents.iloc[c.position]["text"] for c in candidates}
    selected, duplicates, stats = select_evidence(candidates, texts, top_k=5, min_relevance=0.0)
    assert stats["duplicates_collapsed"] == 1  # r4 e r8 têm o mesmo texto
    assert sum(duplicates.values()) == 1


def test_query_processing_extracts_filters():
    pq = process_query("Quais são os principais problemas relatados nas avaliações com nota baixa?", CATS)
    assert (pq.filters.min_score, pq.filters.max_score) == (1, 2)

    pq = process_query("O que os clientes mais elogiam em cama, mesa e banho?", CATS)
    assert (pq.filters.min_score, pq.filters.max_score) == (4, 5)
    assert pq.filters.categories == ["cama_mesa_banho"]

    pq = process_query("Reclamações sobre celulares em 2018", CATS)
    assert pq.filters.categories == ["telefonia"]
    assert (pq.filters.period_start, pq.filters.period_end) == ("2018-01", "2018-12")
    assert pq.filters.max_score == 3

    pq = process_query("Quais padrões nas avaliações de clientes insatisfeitos?", CATS)
    assert pq.filters.max_score == 2

    pq = process_query("Temas das avaliações de pedidos com entrega atrasada", CATS)
    assert pq.filters.delivery_status == ["atrasado"]
    assert "entrega" in pq.topics


def test_query_processing_without_auto_filters():
    pq = process_query("problemas com nota baixa em 2018", CATS, auto_filters=False)
    assert pq.filters.is_empty()


def test_query_processing_does_not_misfire():
    # "presentes" e "valorizam" não devem virar categoria/tema de preço
    pq = process_query("Quais padrões estão presentes nas avaliações? O que os clientes valorizam?", CATS)
    assert pq.filters.categories == []
    assert "preco" not in pq.topics
    # "não recomendo" é negativo, não positivo
    assert process_query("clientes que não recomendam a loja", CATS).sentiment_hint == "negativa"


def test_extract_core_removes_analytical_framing():
    from voc_rag.retrieval.query_processing import extract_core

    core, keys = extract_core("O que os clientes relatam sobre problemas relacionados à entrega?")
    assert core == "problemas entrega"
    assert keys == ["entrega"]  # "problemas" é intenção, não termo-chave

    core, keys = extract_core("Quais são os principais problemas relatados pelos clientes em suas avaliações?")
    assert core == "problemas" and keys == []

    _, keys = extract_core("O que os clientes mais elogiam em cama, mesa e banho?", ["cama_mesa_banho"])
    assert keys == []  # categoria já virou filtro


def test_select_evidence_deprioritizes_short_texts():
    from voc_rag.retrieval.retriever import Candidate

    cands = [Candidate(0, 0.95), Candidate(1, 0.90), Candidate(2, 0.85)]
    texts = {0: "ótimo", 1: "Chegou antes do prazo e bem embalado", 2: "Produto de boa qualidade, recomendo"}
    selected, _, stats = select_evidence(cands, texts, top_k=2, min_relevance=0.5)
    assert [c.position for c in selected] == [1, 2]  # "ótimo" fica de fora
    assert stats["short_texts_deprioritized"] == 1


def test_mmr_prefers_diverse_evidence():
    from voc_rag.retrieval.reranker import _mmr
    from voc_rag.retrieval.retriever import Candidate

    emb = np.array([[1, 0], [0.99, 0.141], [0, 1]], dtype=np.float32)  # 0 e 1 quase iguais
    cands = [Candidate(0, 0.95), Candidate(1, 0.94), Candidate(2, 0.80)]
    chosen = _mmr(cands, emb, k=2, lam=0.5)
    assert [c.position for c in chosen] == [0, 2]
