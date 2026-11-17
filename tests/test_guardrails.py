from voc_rag.config import Settings
from voc_rag.generation.guardrails import EvidenceSignals, assess_evidence, validate_answer
from voc_rag.generation.llm import LLMClient, LLMUnavailableError
from voc_rag.generation.prompts import INSUFFICIENT_MARKER
from voc_rag.pipeline import VoCPipeline
from voc_rag.schemas import AnswerStatus


def signals(**kw):
    base = dict(
        n_candidates=50,
        scope_margin=0.05,
        key_terms=["entrega"],
        unknown_terms=[],
        lexical_support=100,
        relevant_count=20,
        best_similarity=0.9,
        filters_active=False,
    )
    base.update(kw)
    return EvidenceSignals(**base)


def decide(**kw):
    return assess_evidence(signals(**kw), min_evidences=3, scope_margin=0.0, min_relevance=0.85)


def test_allowed_when_all_signals_ok():
    d = decide()
    assert d.allowed and d.status is None


def test_no_candidates_is_insufficient():
    d = decide(n_candidates=0, filters_active=True)
    assert not d.allowed and d.status == AnswerStatus.INSUFFICIENT_EVIDENCE
    assert "filtros" in d.reason


def test_negative_margin_without_lexical_support_is_out_of_scope():
    d = decide(scope_margin=-0.02, lexical_support=0)
    assert d.status == AnswerStatus.OUT_OF_SCOPE


def test_negative_margin_with_strong_lexical_support_is_allowed():
    # ex.: "Recebi tudo certinho": frase afirmativa, mas muito presente na base
    d = decide(scope_margin=-0.015, lexical_support=500)
    assert d.allowed


def test_unknown_terms_is_insufficient():
    d = decide(unknown_terms=["criptomoedas"])
    assert d.status == AnswerStatus.INSUFFICIENT_EVIDENCE and "criptomoedas" in d.reason


def test_low_lexical_support_is_insufficient():
    d = decide(lexical_support=1)
    assert d.status == AnswerStatus.INSUFFICIENT_EVIDENCE


def test_lexical_check_skipped_without_key_terms():
    d = decide(key_terms=[], lexical_support=-1)
    assert d.allowed


def test_few_relevant_is_insufficient():
    d = decide(relevant_count=2)
    assert d.status == AnswerStatus.INSUFFICIENT_EVIDENCE


def test_validate_answer_marker():
    text, status, cited, warnings = validate_answer(f"  {INSUFFICIENT_MARKER} ", {"E1"})
    assert status == AnswerStatus.INSUFFICIENT_EVIDENCE and not cited and warnings


def test_validate_answer_removes_invalid_citations():
    text, status, cited, warnings = validate_answer("Atrasos são comuns [E1][E9].", {"E1", "E2"})
    assert status == AnswerStatus.ANSWERED
    assert cited == {"E1"}
    assert "[E9]" not in text
    assert any("E9" in w for w in warnings)


def test_validate_answer_without_citations_warns():
    _, _, cited, warnings = validate_answer("Resposta sem fontes.", {"E1"})
    assert not cited and warnings


# ----------------------------------------------------------- pipeline (fim a fim)
class EchoLLM(LLMClient):
    """LLM falso: cita as duas primeiras evidências do contexto."""

    def __init__(self):
        super().__init__(api_key="x", model="echo")

    def _call_api(self, system_prompt, user_prompt):
        return "Os clientes relatam atrasos na entrega [E1][E2]."


class StubScope:
    """Classificador de escopo falso: margem negativa só para perguntas com 'Austrália'."""

    def margin(self, question):
        return -0.05 if "Austrália" in question else 0.05


def _pipeline(store, embedder, llm, **kw):
    base = dict(min_relevance=0.3, min_evidences=2, min_term_df=1, use_bm25=True, top_k_context=5)
    base.update(kw)
    return VoCPipeline(store, embedder, llm, Settings(**base), scope_classifier=StubScope())


def test_pipeline_answers_with_citations(store, embedder):
    resp = _pipeline(store, embedder, EchoLLM()).ask("entrega atrasada", auto_filters=False)
    assert resp.status == AnswerStatus.ANSWERED
    assert {e.evidence_id for e in resp.cited_evidences} == {"E1", "E2"}
    assert "<evidencias>" in resp.trace.context_preview


def test_pipeline_retrieval_only_without_llm(store, embedder):
    resp = _pipeline(store, embedder, LLMClient("", "")).ask("entrega atrasada", auto_filters=False)
    assert resp.status == AnswerStatus.RETRIEVAL_ONLY
    assert resp.evidences


def test_pipeline_refuses_out_of_scope(store, embedder):
    resp = _pipeline(store, embedder, EchoLLM()).ask("Qual a capital da Austrália?", auto_filters=False)
    assert resp.status == AnswerStatus.OUT_OF_SCOPE


def test_pipeline_refuses_unknown_concept(store, embedder):
    resp = _pipeline(store, embedder, EchoLLM()).ask("pagamento com criptomoedas", auto_filters=False)
    assert resp.status == AnswerStatus.INSUFFICIENT_EVIDENCE
    assert "criptomoedas" in resp.trace.guardrail["reason"]


def test_pipeline_unconfigured_llm_falls_back(store, embedder):
    resp = _pipeline(store, embedder, LLMClient("", "")).ask("entrega atrasada", auto_filters=False)
    assert resp.status == AnswerStatus.RETRIEVAL_ONLY
    assert "LLM_API_KEY" in resp.answer


class FailingLLM(LLMClient):
    """Simula o Gemini indisponível (ex.: 503 depois dos retries), sem acessar a rede."""

    def __init__(self):
        super().__init__("key", "model")

    def _call_api(self, system_prompt, user_prompt):
        raise LLMUnavailableError("Falha ao chamar o Gemini (503 UNAVAILABLE)")


def test_pipeline_llm_failure_falls_back(store, embedder):
    resp = _pipeline(store, embedder, FailingLLM()).ask("entrega atrasada", auto_filters=False)
    assert resp.status == AnswerStatus.RETRIEVAL_ONLY
    assert "503" in resp.answer
    assert resp.evidences
