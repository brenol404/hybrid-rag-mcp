"""Testes do re-ranking — sem Ollama (stub + fallback puro)."""

from __future__ import annotations

from hybrid_rag_mcp.config import Settings
from hybrid_rag_mcp.models import SearchHit
from hybrid_rag_mcp.providers.rerank import Reranker
from hybrid_rag_mcp.rag.hybrid import hybrid_search
from hybrid_rag_mcp.stores.lexic import LexicalStore
from hybrid_rag_mcp.stores.vector import VectorStore


class _StubVector(VectorStore):
    def __init__(self) -> None:
        pass

    def search(self, query, top_k):
        return [
            SearchHit("a", "doc", "chunk a", 0.99, "vector"),
            SearchHit("b", "doc", "chunk b", 0.98, "vector"),
        ]


class _StubReranker:
    def score(self, query: str, texts: list[str]) -> list[float]:
        return list(range(len(texts)))  # inverte a ordem: último texto recebe nota mais alta


def test_reranker_sem_modelo_retorna_zeros() -> None:
    r = Reranker(Settings(rerank_model=""))
    assert r.score("pergunta", ["a", "b"]) == [0.0, 0.0]


def test_reranker_ignora_endpoint_indisponivel() -> None:
    r = Reranker(Settings(rerank_model="fake-model", ollama_host="http://localhost:1"))
    scores = r.score("x", ["a", "b"])
    assert scores == [0.0, 0.0]


def test_hybrid_rerank_reordena_top_k() -> None:
    reranked = hybrid_search(
        _StubVector(),
        LexicalStore(),
        "pergunta",
        top_k=2,
        bm25_top_k=1,
        reranker=_StubReranker(),  # type: ignore[arg-type]
        rerank_budget=2,
    )
    assert [h.chunk_id for h in reranked] == ["b", "a"]
    assert all(h.strategy == "hybrid" for h in reranked)
