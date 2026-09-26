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


class _FakeHTTPResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


class _FakeHTTPClient:
    def __init__(self, timeout: int | None = None) -> None:
        self.posts: list[dict] = []
        self.closed = False

    def post(self, url: str, headers: dict | None = None, json: dict | None = None):
        self.posts.append({"url": url, "headers": headers, "json": json})
        texts = json["input"] if isinstance(json["input"], list) else [json["input"]]
        # fora de ordem de propósito: o provider deve reordenar por index
        data = [{"index": i, "embedding": [float(i)] * 3} for i in range(len(texts) - 1, -1, -1)]
        return _FakeHTTPResponse({"data": data})

    def close(self) -> None:
        self.closed = True


def _cloud_embedder(monkeypatch):
    from hybrid_rag_mcp.providers.embed import CloudEmbeddings

    fake = _FakeHTTPClient()
    monkeypatch.setattr("httpx.Client", lambda **kwargs: fake)
    settings = Settings(
        cloud_base_url="https://api.exemplo.com/v1",
        cloud_api_key="chave",
        cloud_embed_model="modelo-x",
    )
    return CloudEmbeddings(settings), fake


def test_cloud_embed_post_e_reordena(monkeypatch) -> None:
    emb, fake = _cloud_embedder(monkeypatch)
    vecs = emb.embed(["a", "bb", "ccc"])
    assert len(fake.posts) == 1
    assert fake.posts[0]["url"] == "https://api.exemplo.com/v1/embeddings"
    assert fake.posts[0]["json"]["model"] == "modelo-x"
    assert fake.posts[0]["headers"]["Authorization"] == "Bearer chave"
    assert [v[0] for v in vecs] == [0.0, 1.0, 2.0]
    assert emb.dim == 3


def test_cloud_embed_vazio_nao_chama_rede(monkeypatch) -> None:
    emb, fake = _cloud_embedder(monkeypatch)
    assert emb.embed([]) == []
    assert fake.posts == []


def test_resolve_embedder_escolhe_nuvem_so_completa(monkeypatch) -> None:
    from hybrid_rag_mcp.providers.embed import resolve_embedder

    full = Settings(
        cloud_base_url="https://api.exemplo.com/v1",
        cloud_api_key="chave",
        cloud_embed_model="modelo-x",
    )
    monkeypatch.setattr("httpx.Client", lambda **kwargs: _FakeHTTPClient())
    from hybrid_rag_mcp.providers.embed import CloudEmbeddings, OllamaEmbeddings

    assert isinstance(resolve_embedder(full), CloudEmbeddings)
    partial = Settings(cloud_base_url="https://api.exemplo.com/v1", cloud_api_key="chave")
    assert isinstance(resolve_embedder(partial), OllamaEmbeddings)
    assert isinstance(resolve_embedder(Settings()), OllamaEmbeddings)
