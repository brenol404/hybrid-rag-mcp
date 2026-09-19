from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Gera embeddings para uma lista de textos."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensão dos embeddings gerados."""


@dataclass(frozen=True)
class LLMResponse:
    text: str
    provider: str
    model: str


class LLMProvider(ABC):
    provider_name: str = "abstract"

    @abstractmethod
    def complete(self, system: str, user: str) -> LLMResponse:
        """Gera uma resposta. Levanta exceção se indisponível."""

    def complete_stream(self, system: str, user: str):
        """Gera a resposta em chunks (tokens/trechos). Default: não-streaming."""
        resp = self.complete(system, user)
        yield resp
