from __future__ import annotations

import time
from collections.abc import Iterator

from ..config import Settings
from .base import LLMProvider, LLMResponse


class OllamaLLM(LLMProvider):
    provider_name = "ollama"

    def __init__(self, settings: Settings) -> None:
        import ollama

        self._client = ollama.Client(host=settings.ollama_host)
        self._model = settings.llm_model

    def complete(self, system: str, user: str) -> LLMResponse:
        resp = self._client.chat(
            model=self._model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return LLMResponse(
            text=resp["message"]["content"], provider=self.provider_name, model=self._model
        )

    def complete_stream(self, system: str, user: str) -> Iterator[LLMResponse]:
        resp = self._client.chat(
            model=self._model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            stream=True,
        )
        for chunk in resp:
            delta = chunk["message"]["content"]
            if delta:
                yield LLMResponse(text=delta, provider=self.provider_name, model=self._model)


class CloudLLM(LLMProvider):
    """Cliente OpenAI-compatible (OpenAI, Anthropic via gateway, Together, Groq...)."""

    provider_name = "cloud"

    def __init__(self, settings: Settings) -> None:
        import httpx

        self._base_url = settings.cloud_base_url.rstrip("/")
        self._api_key = settings.cloud_api_key
        self._model = settings.cloud_model
        self._http = httpx.Client(timeout=60)

    def close(self) -> None:
        self._http.close()

    def complete(self, system: str, user: str) -> LLMResponse:
        resp = self._http.post(
            f"{self._base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        return LLMResponse(text=text, provider=self.provider_name, model=self._model)


class FallbackLLM:
    """Tenta o provedor de nuvem e cai automaticamente para Ollama local."""

    def __init__(self, settings: Settings) -> None:
        providers: list[LLMProvider] = [OllamaLLM(settings)]
        if settings.cloud_api_key and settings.cloud_base_url:
            providers.insert(0, CloudLLM(settings))
        self._providers = providers

    @property
    def has_cloud(self) -> bool:
        return len(self._providers) > 1 and self._providers[0].provider_name == "cloud"

    def close(self) -> None:
        """Fecha os provedores que seguram recursos (ex.: httpx.Client da nuvem)."""
        for provider in self._providers:
            close = getattr(provider, "close", None)
            if callable(close):
                close()

    def complete(self, system: str, user: str) -> LLMResponse:
        errors: list[str] = []
        for provider in self._providers:
            try:
                return provider.complete(system, user)
            except Exception as exc:  # noqa: BLE001 - fallback deve capturar qualquer falha de provedor
                errors.append(f"{provider.provider_name}: {exc}")
                time.sleep(0.2)
        raise RuntimeError(f"Todos os provedores de LLM falharam: {'; '.join(errors)}")

    def complete_stream(self, system: str, user: str) -> Iterator[LLMResponse]:
        """Streaming com fallback entre provedores: o primeiro que emitir vence."""
        errors: list[str] = []
        for provider in self._providers:
            gen = provider.complete_stream(system, user)
            try:
                first = next(gen)
            except StopIteration:
                errors.append(f"{provider.provider_name}: resposta vazia")
                continue
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{provider.provider_name}: {exc}")
                time.sleep(0.2)
                continue
            yield first
            yield from gen
            return
        raise RuntimeError(f"Todos os provedores de LLM falharam ao gerar: {'; '.join(errors)}")
