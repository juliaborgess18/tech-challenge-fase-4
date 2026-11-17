"""Cliente do modelo de linguagem (geração da resposta).

Provedor: Google Gemini (SDK `google-genai`).

1. Instale o SDK: `pip install google-genai` (ou `pip install -e ".[llm]"`).
2. Preencha LLM_API_KEY e LLM_MODEL (ex.: gemini-3.5-flash-lite) no arquivo .env.

Enquanto a chave não estiver configurada, o pipeline funciona em modo
"somente recuperação" (status RETRIEVAL_ONLY): devolve as evidências
recuperadas, sem texto gerado.
"""

from __future__ import annotations


class LLMNotConfiguredError(RuntimeError):
    """O LLM não foi configurado (sem chave ou sem SDK instalado)."""


class LLMUnavailableError(LLMNotConfiguredError):
    """A chamada ao LLM falhou (chave inválida, modelo indisponível, timeout).

    Herda de LLMNotConfiguredError para que o pipeline caia no modo
    "somente recuperação" em vez de quebrar a interface.
    """


class LLMClient:
    def __init__(self, api_key: str, model: str, max_tokens: int = 4000, thinking_budget: int = 0):
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        # 0 = sem raciocínio prévio (mais rápido); -1 = o modelo decide.
        # Modelos sem esse recurso (ex.: flash-lite) recusam a opção: aí ela é desativada.
        self.thinking_budget = thinking_budget
        self._thinking_supported = True
        self._client = None  # reaproveitado entre chamadas (evita nova conexão a cada pergunta)

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.model)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Gera a resposta a partir do prompt de sistema e do prompt com as evidências."""
        if not self.is_configured:
            raise LLMNotConfiguredError(
                "Defina LLM_API_KEY e LLM_MODEL no arquivo .env (local) ou nos Secrets do Streamlit Cloud."
            )
        return self._call_api(system_prompt, user_prompt)

    def _call_api(self, system_prompt: str, user_prompt: str) -> str:
        # Google Gemini via SDK oficial (`pip install google-genai`).
        try:
            from google import genai
            from google.genai import errors, types
        except ImportError as exc:
            raise LLMNotConfiguredError(
                "SDK do Gemini não instalado: rode `pip install google-genai`."
            ) from exc

        # Retry curto para erros transitórios (429 limite de taxa, 5xx sobrecarga) e
        # timeout de 60 s por tentativa, para a interface não ficar presa por minutos.
        if self._client is None:
            self._client = genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(
                    timeout=60_000,
                    retry_options=types.HttpRetryOptions(
                        attempts=3, max_delay=10, http_status_codes=[429, 500, 502, 503, 504]
                    ),
                ),
            )

        def request(with_thinking: bool):
            thinking = (
                types.ThinkingConfig(thinking_budget=self.thinking_budget)
                if with_thinking and self.thinking_budget >= 0
                else None
            )
            return self._client.models.generate_content(
                model=self.model,  # ex.: "gemini-3.5-flash-lite"
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=self.max_tokens,
                    thinking_config=thinking,
                ),
            )

        try:
            try:
                response = request(with_thinking=self._thinking_supported)
            except errors.ClientError as exc:
                # modelo sem suporte à configuração de raciocínio: repete sem ela e memoriza
                if exc.code != 400 or not self._thinking_supported or self.thinking_budget < 0:
                    raise
                self._thinking_supported = False
                response = request(with_thinking=False)
        except errors.APIError as exc:
            raise LLMUnavailableError(
                f"Falha ao chamar o Gemini ({exc.code} {exc.status}): {exc.message}; "
                "exibindo apenas as avaliações recuperadas."
            ) from exc
        except Exception as exc:  # timeout / rede
            raise LLMUnavailableError(
                f"Falha de conexão com o Gemini ({type(exc).__name__}); "
                "exibindo apenas as avaliações recuperadas."
            ) from exc

        text = response.text
        if not text:
            reason = response.candidates[0].finish_reason if response.candidates else "desconhecido"
            raise LLMUnavailableError(
                f"O Gemini não retornou texto (finish_reason={reason}); "
                "exibindo apenas as avaliações recuperadas."
            )
        return text
