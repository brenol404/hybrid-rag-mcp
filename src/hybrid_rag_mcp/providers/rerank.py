from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import httpx

from ..config import Settings

logger = logging.getLogger("hybrid_rag_mcp.providers.rerank")


class BaseReranker(ABC):
    """Interface base para provedores de re-ranking (Cross-Encoders)."""

    @abstractmethod
    def score(self, query: str, texts: list[str]) -> list[float]:
        """Calcula scores de relevância para a lista de textos em relação à query."""
        raise NotImplementedError

    def close(self) -> None:
        """Libera conexões HTTP ou recursos."""


class OllamaReranker(BaseReranker):
    """Cross-encoder via endpoint /api/rerank do Ollama (versão >= 0.36, ex.: bge-reranker-v2-m3).

    Degradação graciosa: se o modelo não estiver disponível ou o endpoint falhar,
    retorna 0.0 para todos os itens sem interromper o pipeline.
    """

    def __init__(self, settings: Settings) -> None:
        self._model = settings.rerank_model
        self._url = settings.rerank_url or f"{settings.ollama_host.rstrip('/')}/api/rerank"
        self._http = httpx.Client(timeout=60)

    def close(self) -> None:
        self._http.close()

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not self._model or not texts:
            return [0.0] * len(texts)
        try:
            resp = self._http.post(
                self._url,
                json={"model": self._model, "query": query, "documents": texts},
            )
            if resp.status_code >= 400:
                logger.warning("Ollama rerank retornou status %s", resp.status_code)
                return [0.0] * len(texts)
            results = resp.json().get("results", [])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Falha na chamada de rerank Ollama: %s", exc)
            return [0.0] * len(texts)

        ranked = [(r["index"], r.get("relevance_score", 0.0)) for r in results]
        scores = [0.0] * len(texts)
        for index, score in ranked:
            if 0 <= index < len(texts):
                scores[index] = float(score)
        return scores


class CohereReranker(BaseReranker):
    """Re-ranking neural via API da Cohere (ex: rerank-v3.5, rerank-multilingual-v3.0)."""

    COHERE_API_URL = "https://api.cohere.com/v2/rerank"

    def __init__(self, settings: Settings) -> None:
        self._api_key = settings.cohere_api_key
        self._model = settings.rerank_model or "rerank-v3.5"
        self._url = settings.rerank_url or self.COHERE_API_URL
        self._http = httpx.Client(timeout=60)

    def close(self) -> None:
        self._http.close()

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not self._api_key or not texts:
            return [0.0] * len(texts)
        try:
            resp = self._http.post(
                self._url,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model,
                    "query": query,
                    "documents": texts,
                    "top_n": len(texts),
                },
            )
            if resp.status_code >= 400:
                logger.warning("Cohere rerank retornou status %s", resp.status_code)
                return [0.0] * len(texts)
            results = resp.json().get("results", [])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Falha na chamada de rerank Cohere: %s", exc)
            return [0.0] * len(texts)

        ranked = [(r["index"], r.get("relevance_score", 0.0)) for r in results]
        scores = [0.0] * len(texts)
        for index, score in ranked:
            if 0 <= index < len(texts):
                scores[index] = float(score)
        return scores


class CustomHttpReranker(BaseReranker):
    """Re-ranking via endpoint HTTP customizado (ex: Text-Embeddings-Inference ou FastAPI)."""

    def __init__(self, settings: Settings) -> None:
        self._url = settings.rerank_url
        self._model = settings.rerank_model
        self._http = httpx.Client(timeout=60)

    def close(self) -> None:
        self._http.close()

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not self._url or not texts:
            return [0.0] * len(texts)
        try:
            payload: dict[str, Any] = {"query": query, "documents": texts}
            if self._model:
                payload["model"] = self._model
            resp = self._http.post(self._url, json=payload)
            if resp.status_code >= 400:
                return [0.0] * len(texts)
            data = resp.json()
            if isinstance(data, list):
                # Formato lista direta de scores: [0.95, 0.82, ...]
                return [float(s) for s in data[: len(texts)]]
            results = data.get("results", [])
            scores = [0.0] * len(texts)
            for r in results:
                idx = r.get("index", 0)
                if 0 <= idx < len(texts):
                    scores[idx] = float(r.get("relevance_score", r.get("score", 0.0)))
            return scores
        except Exception as exc:  # noqa: BLE001
            logger.warning("Falha no reranker customizado: %s", exc)
            return [0.0] * len(texts)


# Alias para retrocompatibilidade
Reranker = OllamaReranker


def build_reranker(settings: Settings) -> BaseReranker | None:
    provider = (settings.rerank_provider or "").lower().strip()
    if provider == "cohere" or (settings.cohere_api_key and not settings.rerank_model):
        return CohereReranker(settings)
    if provider == "custom" or (settings.rerank_url and provider != "ollama"):
        return CustomHttpReranker(settings)
    if settings.rerank_model:
        return OllamaReranker(settings)
    return None
