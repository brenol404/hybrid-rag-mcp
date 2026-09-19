from __future__ import annotations

from typing import Literal

from ..config import Settings
from .base import EmbeddingProvider


class OllamaEmbeddings(EmbeddingProvider):
    def __init__(self, settings: Settings) -> None:
        import ollama

        self._client = ollama.Client(host=settings.ollama_host)
        self._model = settings.embed_model

    @property
    def dim(self) -> int:
        return len(self.embed(["p"])[0])

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._client.embed(model=self._model, input=t)["embeddings"][0] for t in texts]


def resolve_embedder(settings: Settings, mode: Literal["local"] = "local") -> EmbeddingProvider:
    return OllamaEmbeddings(settings)
