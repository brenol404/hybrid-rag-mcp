"""Testes do embedder Ollama — batching e dim preguiçosa, sem rede (cliente fake)."""

from __future__ import annotations

from hybrid_rag_mcp.config import Settings
from hybrid_rag_mcp.providers.embed import OllamaEmbeddings


class _FakeOllamaClient:
    def __init__(self, host: str | None = None) -> None:
        self.calls: list[list[str]] = []

    def embed(self, model: str | None = None, input=None):
        batch = list(input) if isinstance(input, list) else [input]
        self.calls.append(batch)
        return {"embeddings": [[float(len(t))] * 4 for t in batch]}


def _embedder(monkeypatch, **overrides) -> tuple[OllamaEmbeddings, _FakeOllamaClient]:
    fakes: list[_FakeOllamaClient] = []

    def factory(host: str | None = None) -> _FakeOllamaClient:
        fake = _FakeOllamaClient(host)
        fakes.append(fake)
        return fake  # type: ignore[return-value]

    monkeypatch.setattr("ollama.Client", factory)
    return OllamaEmbeddings(Settings(**overrides)), fakes[0]


def test_embed_lote_preserva_ordem(monkeypatch) -> None:
    emb, fake = _embedder(monkeypatch, embed_batch_size=32)
    texts = [f"x{i}" for i in range(70)]
    vecs = emb.embed(texts)
    assert [len(c) for c in fake.calls] == [32, 32, 6]
    assert [v[0] for v in vecs] == [float(len(t)) for t in texts]


def test_embed_vazio_nao_chama_rede(monkeypatch) -> None:
    emb, fake = _embedder(monkeypatch)
    assert emb.embed([]) == []
    assert fake.calls == []


def test_dim_resolve_uma_vez_e_cacheia(monkeypatch) -> None:
    emb, fake = _embedder(monkeypatch)
    assert emb.dim == 4
    assert emb.dim == 4
    assert len(fake.calls) == 1
