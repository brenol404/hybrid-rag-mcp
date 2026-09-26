from __future__ import annotations

import json
import time
from collections.abc import Iterator
from typing import Any

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

    def __init__(
        self,
        settings: Settings,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        import httpx

        self._base_url = (base_url or settings.cloud_base_url).rstrip("/")
        self._api_key = api_key or settings.cloud_api_key
        self._model = model or settings.cloud_model
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


def parse_llm_chain(settings: Settings) -> list[dict[str, Any]]:
    """Lista ordenada de endpoints de nuvem a partir da config.

    `LLM_CHAIN` (JSON) vence; sem ele, o trio `CLOUD_*` vira 1 entrada;
    sem nada, lista vazia (só Ollama). Erro claro em JSON/forma inválidos.
    """
    raw = settings.llm_chain.strip()
    if raw:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM_CHAIN com JSON inválido: {exc}") from exc
        if not isinstance(data, list):
            raise ValueError("LLM_CHAIN deve ser uma lista JSON de {base_url, api_key, model}")
        entries = data
    elif settings.cloud_base_url and settings.cloud_api_key:
        entries = [
            {
                "base_url": settings.cloud_base_url,
                "api_key": settings.cloud_api_key,
                "model": settings.cloud_model,
            }
        ]
    else:
        return []
    required = ("base_url", "api_key", "model")
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict) or any(not entry.get(k) for k in required):
            raise ValueError(
                f"LLM_CHAIN[{i}] inválido: cada elo precisa de {list(required)} preenchidos"
            )
    return entries


class FallbackLLM:
    """Cascata de provedores em ordem de prioridade; Ollama local sempre por último.

    Fontes da lista: `LLM_CHAIN` (JSON) ou o trio `CLOUD_*` (compat). Sem nuvem,
    vira só Ollama. A primeira resposta válida vence; falhas caem para o próximo.
    """

    def __init__(
        self, settings: Settings | None = None, providers: list[LLMProvider] | None = None
    ) -> None:
        if providers is not None:
            self._providers = providers
            return
        assert settings is not None
        chain: list[LLMProvider] = [
            CloudLLM(
                settings,
                base_url=entry["base_url"],
                api_key=entry["api_key"],
                model=entry["model"],
            )
            for entry in parse_llm_chain(settings)
        ]
        chain.append(OllamaLLM(settings))
        self._providers = chain

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
