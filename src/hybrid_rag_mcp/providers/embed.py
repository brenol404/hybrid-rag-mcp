from __future__ import annotations

from typing import Literal

from ..config import Settings
from .base import EmbeddingProvider


class OllamaEmbeddings(EmbeddingProvider):
    def __init__(self, settings: Settings) -> None:
        import ollama

        self._client = ollama.Client(host=settings.ollama_host)
        self._model = settings.embed_model
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
        return [self._client.embed(model=self._model, input=t)["embeddings"][0] for t in texts]


def resolve_embedder(settings: Settings, mode: Literal["local"] = "local") -> EmbeddingProvider:
    return OllamaEmbeddings(settings)
