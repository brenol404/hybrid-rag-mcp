"""Integração: cache semântico e compressão dentro do RAGEngine — sem Ollama nem Qdrant."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hybrid_rag_mcp.config import Settings
from hybrid_rag_mcp.models import SearchHit
from hybrid_rag_mcp.providers.base import EmbeddingProvider, LLMResponse
from hybrid_rag_mcp.rag import engine as engine_module
from hybrid_rag_mcp.rag.engine import RAGEngine

_DIM = 8


class _FakeEmbed(EmbeddingProvider):
    @property
    def dim(self) -> int:
        return _DIM

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * _DIM
            for word in text.lower().split():
                vec[ord(word[0]) % _DIM] += 1.0
            out.append(vec)
        return out


class _FakeVector:
    def __init__(self, settings=None, embedder=None) -> None:
        pass

    def all_chunks(self) -> list:
        return []

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        return [
            SearchHit(
                "c1",
                "manual.pdf",
                "A janela de manutenção é das 02h às 04h.",
                0.9,
                "vector",
            )
        ]

    def close(self) -> None:
        pass


class _FakeLexical:
    def search(self, query: str, top_k: int) -> list[SearchHit]:
        return []


class _FakeLLM:
    def __init__(self, settings=None) -> None:
        self.calls = 0
        self.last_user = ""

    def complete(self, system: str, user: str) -> LLMResponse:
        self.calls += 1
        self.last_user = user
        return LLMResponse("A janela é 02h-04h [manual.pdf].", "ollama", "qwen3")

    def complete_stream(self, system: str, user: str):
        yield self.complete(system, user)


def _patch(monkeypatch) -> None:
    monkeypatch.setattr(engine_module, "resolve_embedder", lambda settings: _FakeEmbed())
    monkeypatch.setattr(engine_module, "VectorStore", _FakeVector)
    monkeypatch.setattr(engine_module, "LexicalStore", _FakeLexical)
    monkeypatch.setattr(engine_module, "FallbackLLM", _FakeLLM)
    monkeypatch.setattr(engine_module, "build_reranker", lambda settings: None)


def _settings(tmp_path: Path, **overrides) -> Settings:
    base = {
        "qdrant_path": str(tmp_path / "qdrant"),
        "cache_path": str(tmp_path / "cache.jsonl"),
        "audit_log": str(tmp_path / "audit.jsonl"),
        "cache_enabled": True,
        "context_compression": 0,
        "ask_max_iterations": 1,
    }
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
def engine(tmp_path: Path, monkeypatch) -> RAGEngine:
    _patch(monkeypatch)
    return RAGEngine(_settings(tmp_path))


def test_miss_then_hit_skips_second_generation(engine: RAGEngine) -> None:
    first = engine.ask("qual é a janela de manutenção?")
    assert first.cache_hit is False
    assert first.iterations == 1

    same = engine.ask("QUAL é a janela  de manutenção?")
    assert same.cache_hit is True
    assert same.iterations == 0
    assert same.answer == first.answer

    assert engine._llm.calls == 1


def test_distinct_question_regenerates(engine: RAGEngine) -> None:
    engine.ask("qual é a janela de manutenção?")
    engine.ask("quanto custa o plano enterprise?")
    assert engine._llm.calls == 2


def test_cache_hit_registra_no_audit(engine: RAGEngine, tmp_path: Path) -> None:
    engine.ask("janela de manutenção")
    engine.ask("janela de manutenção")
    lines = (tmp_path / "audit.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["cache"] is False
    assert json.loads(lines[1])["cache"] is True


def test_engine_aplica_compressao_e_preserva_fontes(tmp_path: Path, monkeypatch) -> None:
    _patch(monkeypatch)
    eng = RAGEngine(
        _settings(
            tmp_path,
            cache_enabled=False,
            context_compression=2,
        )
    )
    result = eng.ask("janela de manutenção")
    # nível 2 remove determinantes/auxiliares apenas na CÓPIA do prompt (user)
    assert "é das 02h às 04h" not in eng._llm.last_user
    assert "02h às 04h" in eng._llm.last_user
    # fontes preservam o texto original, sem compactação
    assert "A janela de manutenção é das 02h às 04h." == result.sources[0].content


def test_engine_init_nao_consulta_embedder(tmp_path: Path, monkeypatch) -> None:
    """Construir o RAGEngine não toca no Ollama/Qdrant: dim é preguiçosa.

    Com o VectorStore REAL (Qdrant local), o construtor não deve chamar
    `embed` nem criar a coleção — isso fica para o primeiro search/ingest.
    """

    class _CountingEmbed(_FakeEmbed):
        def __init__(self) -> None:
            self.calls = 0

        def embed(self, texts: list[str]) -> list[list[float]]:
            self.calls += 1
            return super().embed(texts)

    counter = _CountingEmbed()
    monkeypatch.setattr(engine_module, "resolve_embedder", lambda settings: counter)
    monkeypatch.setattr(engine_module, "build_reranker", lambda settings: None)
    monkeypatch.setattr(engine_module, "FallbackLLM", _FakeLLM)

    RAGEngine(_settings(tmp_path))
    assert counter.calls == 0


def test_audit_rotaciona_por_tamanho(tmp_path: Path, monkeypatch) -> None:
    """Higiene de dados: audit.jsonl não cresce sem limite (rotação .1, .2, ...)."""
    _patch(monkeypatch)
    eng = RAGEngine(_settings(tmp_path, cache_enabled=False, audit_max_bytes=400, audit_keep=2))
    for i in range(6):
        eng.ask(f"pergunta numero {i} sobre a janela de manutencao?")
    audit = tmp_path / "audit.jsonl"
    assert (tmp_path / "audit.jsonl.1").exists()
    assert audit.exists()
    # nada se perde: atual + backups somam as 6 linhas
    total = 0
    for f in (audit, tmp_path / "audit.jsonl.1", tmp_path / "audit.jsonl.2"):
        if f.exists():
            total += len(f.read_text(encoding="utf-8").strip().splitlines())
    assert total == 6
