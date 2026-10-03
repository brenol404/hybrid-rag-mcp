from __future__ import annotations

from pathlib import Path

import pytest
from starlette.testclient import TestClient

from hybrid_rag_mcp.config import Settings
from hybrid_rag_mcp.providers.base import EmbeddingProvider, LLMResponse
from hybrid_rag_mcp.rag import engine as engine_module
from hybrid_rag_mcp.rag.engine import RAGEngine
from hybrid_rag_mcp.ui import UIApp

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


class _FakeLLM:
    def __init__(self, settings=None) -> None:
        self.calls = 0

    def complete(self, system: str, user: str) -> LLMResponse:
        self.calls += 1
        return LLMResponse("O RAG híbrido combina vetores com BM25 [guia-rag.md].", "mock", "test-model")

    def complete_stream(self, system: str, user: str):
        yield self.complete(system, user)

    def close(self) -> None:
        pass


def test_ui_endpoints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine_module, "resolve_embedder", lambda settings: _FakeEmbed())
    monkeypatch.setattr(engine_module, "FallbackLLM", _FakeLLM)

    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        cache_enabled=False,
    )
    engine = RAGEngine(settings)
    ui_app = UIApp(engine=engine, settings=settings)
    app = ui_app.create_app()
    client = TestClient(app)

    # 1. Health check
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}

    # 2. Index HTML page
    res = client.get("/")
    assert res.status_code == 200
    assert "Hybrid RAG MCP" in res.text
    assert "Live Inspector" in res.text

    # 3. Stats endpoint before ingestion
    res = client.get("/api/stats")
    assert res.status_code == 200
    stats = res.json()
    assert stats["vector_chunks"] == 0
    assert stats["lexical_chunks"] == 0
    assert stats["dim"] == _DIM

    # 4. Ingest text
    res = client.post(
        "/api/ingest_text",
        json={
            "doc_name": "guia-rag.md",
            "content": "# Arquitetura RAG Híbrida\nO RAG híbrido combina vetores semânticos com BM25.",
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["chunks"] >= 1
    assert data["indexed"] >= 1

    # Ingest text validation error
    res_bad = client.post("/api/ingest_text", json={"doc_name": "", "content": ""})
    assert res_bad.status_code == 400

    # 5. List chunks
    res = client.get("/api/chunks")
    assert res.status_code == 200
    chunks = res.json()
    assert len(chunks) >= 1
    assert chunks[0]["doc_name"] == "guia-rag.md"

    # Filter chunks by doc
    res_filtered = client.get("/api/chunks?doc=guia")
    assert res_filtered.status_code == 200
    assert len(res_filtered.json()) >= 1

    # 6. Detailed search
    res = client.post(
        "/api/search",
        json={"query": "RAG híbrido vetores", "top_k": 3, "weights": [1.0, 1.0]},
    )
    assert res.status_code == 200
    search_data = res.json()
    assert "hybrid" in search_data
    assert "vector" in search_data
    assert "lexical" in search_data
    assert len(search_data["hybrid"]) >= 1

    # Search validation error
    res_empty = client.post("/api/search", json={"query": ""})
    assert res_empty.status_code == 400

    # 7. Ask endpoint
    res = client.post("/api/ask", json={"question": "O que combina o RAG híbrido?"})
    assert res.status_code == 200
    ask_data = res.json()
    assert "answer" in ask_data
    assert "trace" in ask_data
    assert ask_data["provider"] == "mock"

    # Ask validation error
    res_bad_ask = client.post("/api/ask", json={"question": ""})
    assert res_bad_ask.status_code == 400

    # 8. Stats endpoint after ingestion
    res = client.get("/api/stats")
    assert res.status_code == 200
    assert res.json()["vector_chunks"] >= 1

    engine.close()
