from __future__ import annotations

from typing import Literal

from ..config import Settings
from .base import EmbeddingProvider


class OllamaEmbeddings(EmbeddingProvider):
    def __init__(self, settings: Settings) -> None:
        import ollama

        self._client = ollama.Client(host=settings.ollama_host)
        self._model = settings.embed_model
        self._batch_size = max(1, settings.embed_batch_size)
        self._dim_cache: int | None = None

    @property
    def dim(self) -> int:
        """Dimensão resolvida sob demanda (1ª chamada) e cacheada depois.

        Antes, o construtor do engine consultava o Ollama para saber a dim e
        criar a coleção — qualquer falha de startup derrubava o primeiro uso.
        """
        if self._dim_cache is None:
            self._dim_cache = len(self.embed(["p"])[0])
        return self._dim_cache

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed em lotes (1 request por lote) preservando a ordem de entrada."""
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            resp = self._client.embed(model=self._model, input=batch)
            out.extend(resp["embeddings"])
        return out


class CloudEmbeddings(EmbeddingProvider):
    """Embeddings via API OpenAI-compatível (`POST {base}/embeddings`).

    Opt-in: exige CLOUD_BASE_URL + CLOUD_API_KEY + CLOUD_EMBED_MODEL. Trocar o
    modelo de embedding muda a dimensão → o índice precisa ser recriado
    (ver QDRANT_RECREATE_ON_DIM_CHANGE) e tudo re-embedado via `ingest`.
    """

    def __init__(self, settings: Settings) -> None:
        import httpx

        self._url = f"{settings.cloud_base_url.rstrip('/')}/embeddings"
        self._api_key = settings.cloud_api_key
        self._model = settings.cloud_embed_model
        self._http = httpx.Client(timeout=120)
        self._dim_cache: int | None = None

    @property
    def dim(self) -> int:
        if self._dim_cache is None:
            self._dim_cache = len(self.embed(["p"])[0])
        return self._dim_cache

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = self._http.post(
            self._url,
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={"model": self._model, "input": texts},
        )
        resp.raise_for_status()
        data = sorted(resp.json()["data"], key=lambda d: d["index"])
        return [list(map(float, d["embedding"])) for d in data]

    def close(self) -> None:
        self._http.close()


def resolve_embedder(settings: Settings, mode: Literal["local"] = "local") -> EmbeddingProvider:
    """Ollama por default; nuvem quando as 3 vars estão setadas (opt-in)."""
    if settings.cloud_base_url and settings.cloud_api_key and settings.cloud_embed_model:
        return CloudEmbeddings(settings)
    return OllamaEmbeddings(settings)
