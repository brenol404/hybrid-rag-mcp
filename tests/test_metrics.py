"""Testes de observabilidade — registry, engine instrumentado e tool MCP, sem rede."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from pathlib import Path

from hybrid_rag_mcp.config import Settings
from hybrid_rag_mcp.metrics import Metrics
from hybrid_rag_mcp.models import SearchHit
from hybrid_rag_mcp.providers.base import EmbeddingProvider, LLMResponse
from hybrid_rag_mcp.rag import engine as engine_module
from hybrid_rag_mcp.rag.engine import RAGEngine


def test_contadores_e_latencias() -> None:
    m = Metrics()
    m.inc("search_total")
    m.inc("search_total")
    m.inc("search_errors")
    for v in (10.0, 20.0, 30.0):
        m.observe("search_ms", v)
    snap = m.snapshot()
    assert snap["counters"] == {"search_total": 2, "search_errors": 1}
    lat = snap["latency_ms"]["search_ms"]
    assert lat["n"] == 3 and lat["mean"] == 20.0
    assert lat["p50"] == 20.0 and lat["p95"] == 30.0


def test_render_prometheus_formato() -> None:
    m = Metrics()
    m.inc("ask_total")
    m.observe("ask_ms", 100.0)
    out = m.render_prometheus()
    assert "rag_ask_total 1" in out
    assert 'rag_latency_ms{op="ask_ms",quantile="0.5"} 100.0' in out
    assert 'rag_latency_ms_count{op="ask_ms"} 1' in out


def test_render_text_resumo() -> None:
    m = Metrics()
    m.inc("ask_cache_hits")
    m.observe("search_ms", 50.0)
    out = m.render_text()
    assert "ask_cache_hits: 1" in out
    assert "search_ms" in out and "p95=" in out


def test_thread_safe_sob_concorrencia() -> None:
    m = Metrics()
    threads = [
        threading.Thread(target=lambda: [m.inc("x"), m.observe("y_ms", 1.0)][0]) for _ in range(20)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert m.snapshot()["counters"] == {"x": 20}
    assert m.snapshot()["latency_ms"]["y_ms"]["n"] == 20


class _FakeEmbed(EmbeddingProvider):
    @property
    def dim(self) -> int:
        return 4

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


class _FakeVector:
    def __init__(self, settings=None, embedder=None) -> None:
        pass

    def all_chunks(self) -> list:
        return []

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        return [SearchHit("c1", "doc.pdf", "conteúdo", 0.9, "vector")]

    def close(self) -> None:
        pass


class _FakeLexical:
    def search(self, query: str, top_k: int) -> list[SearchHit]:
        return []


class _FakeLLM:
    def __init__(self, settings=None) -> None:
        pass

    def complete(self, system: str, user: str) -> LLMResponse:
        return LLMResponse("resposta [doc.pdf].", "ollama", "qwen3")

    def complete_stream(self, system: str, user: str):
        yield self.complete(system, user)


def _engine(tmp_path: Path, monkeypatch) -> RAGEngine:
    monkeypatch.setattr(engine_module, "resolve_embedder", lambda settings: _FakeEmbed())
    monkeypatch.setattr(engine_module, "VectorStore", _FakeVector)
    monkeypatch.setattr(engine_module, "LexicalStore", _FakeLexical)
    monkeypatch.setattr(engine_module, "FallbackLLM", _FakeLLM)
    monkeypatch.setattr(engine_module, "build_reranker", lambda settings: None)
    return RAGEngine(
        Settings(
            qdrant_path=str(tmp_path / "qdrant"),
            cache_path=str(tmp_path / "cache.jsonl"),
            audit_log=str(tmp_path / "audit.jsonl"),
            ask_max_iterations=1,
        )
    )


def test_engine_conta_buscas_asks_e_cache(tmp_path: Path, monkeypatch) -> None:
    eng = _engine(tmp_path, monkeypatch)
    eng.search("algo")
    eng.ask("pergunta?")
    eng.ask("pergunta?")  # hit
    counters = eng.metrics.snapshot()["counters"]
    assert counters["search_total"] == 1
    assert counters["ask_total"] == 2
    assert counters["ask_cache_hits"] == 1
    assert counters.get("search_errors", 0) == 0
    assert eng.metrics.snapshot()["latency_ms"]["ask_ms"]["n"] == 2


def test_log_json_estruturado(tmp_path: Path, monkeypatch, caplog) -> None:
    eng = _engine(tmp_path, monkeypatch)
    with caplog.at_level(logging.INFO, logger="hybrid_rag_mcp.engine"):
        eng.search("algo")
    ops = [json.loads(r.message) for r in caplog.records if r.message.startswith("{")]
    assert ops and ops[0]["op"] == "search"
    assert ops[0]["hits"] == 1 and "elapsed_ms" in ops[0]


def test_tool_metrics_registrada() -> None:
    from hybrid_rag_mcp.server import mcp

    tools = asyncio.run(mcp.list_tools())
    assert {t.name for t in tools} == {"ask", "ingest", "metrics", "search"}
