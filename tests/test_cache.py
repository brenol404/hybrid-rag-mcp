"""Testes do cache semântico — embedder stub determinístico, tmp_path."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hybrid_rag_mcp.config import Settings
from hybrid_rag_mcp.models import SearchHit
from hybrid_rag_mcp.optimize import SemanticCache
from hybrid_rag_mcp.providers.base import EmbeddingProvider


class FakeEmbed(EmbeddingProvider):
    """Embedding determinístico por palavra: tokens iguais → vetores iguais."""

    _DIM = 8

    @property
    def dim(self) -> int:
        return self._DIM

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    @classmethod
    def _vec(cls, text: str) -> list[float]:
        vec = [0.0] * cls._DIM
        for word in text.lower().split():
            vec[ord(word[0]) % cls._DIM] += 1.0
        return vec


def _settings(tmp_path: Path, **overrides) -> Settings:
    base = {
        "cache_path": str(tmp_path / "cache.jsonl"),
        "qdrant_path": str(tmp_path / "qdrant"),
        "cache_enabled": True,
        "cache_sim_threshold": 0.92,
        "cache_ttl_sec": 3600,
        "cache_max_entries": 3,
    }
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
def cache(tmp_path: Path) -> SemanticCache:
    return SemanticCache(_settings(tmp_path), FakeEmbed())


def _sources() -> list[SearchHit]:
    return [SearchHit("c1", "manual.pdf", "trecho do manual", 0.9, "hybrid")]


def test_miss_returns_none(cache: SemanticCache) -> None:
    assert cache.lookup("qual a janela de manutenção?") is None


def test_exact_repeat_is_cached(cache: SemanticCache) -> None:
    q = "qual a janela de manutenção?"
    cache.store(q, "resposta", "ollama", "qwen3", _sources())
    hit = cache.lookup(" QUAL   A janela de manutenção? ")
    assert hit is not None
    assert hit.answer == "resposta"
    assert hit.provider == "ollama" and hit.model == "qwen3"
    assert hit.sources[0].doc_name == "manual.pdf"


def test_semantic_similarity_hits_within_threshold(cache: SemanticCache) -> None:
    cache.store("servidor usa redis", "resposta", "ollama", "qwen3", _sources())
    # mesma bag-of-words em ordem diferente: cosseno 1.0, texto normalizado difere →
    # só o caminho semântico (embedding) consegue devolver
    hit = cache.lookup("redis usa servidor")
    assert hit is not None and hit.similarity >= 0.999


def test_distinct_question_misses(cache: SemanticCache) -> None:
    cache.store("o servidor usa redis", "resposta", "ollama", "qwen3", _sources())
    assert cache.lookup("como calcular impostos sobre vendas") is None


def test_ttl_expiry(monkeypatch, cache: SemanticCache) -> None:
    cache.store("pergunta qualquer", "resposta", "ollama", "qwen3", _sources())
    now = [time.time()]
    monkeypatch.setattr("hybrid_rag_mcp.optimize.cache.time.time", lambda: now[0])
    hit = cache.lookup("pergunta qualquer")
    assert hit is not None
    now[0] += 10_000  # > ttl de 3600s
    assert cache.lookup("pergunta qualquer") is None


def test_cache_is_bounded(cache: SemanticCache, tmp_path: Path) -> None:
    for word in ("gold", "blue", "cyan", "magenta", "lime"):
        cache.store(word, f"resposta {word}", "ollama", "qwen3", _sources())
    assert cache.size <= 3
    # resposta mais antiga saiu (gold); a última está lá (lime)
    assert cache.lookup("lime") is not None
    assert cache.lookup("gold") is None


def test_persists_across_instances(cache: SemanticCache, tmp_path: Path) -> None:
    cache.store("duradoura", "resposta", "ollama", "qwen3", _sources())
    reloaded = SemanticCache(_settings(tmp_path), FakeEmbed())
    hit = reloaded.lookup("duradoura")
    assert hit is not None and hit.answer == "resposta"


def test_clear_empties(cache: SemanticCache) -> None:
    cache.store("x", "y", "ollama", "qwen3", _sources())
    assert cache.size == 1
    cache.clear()
    assert cache.size == 0
    assert cache.lookup("x") is None
