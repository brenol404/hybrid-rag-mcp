"""Testes de persistência — Qdrant + BM25 sem necessidade de Ollama (embedder stub)."""

from __future__ import annotations

from pathlib import Path

from hybrid_rag_mcp.config import Settings
from hybrid_rag_mcp.rag.hybrid import hybrid_search
from hybrid_rag_mcp.rag.ingestion import ingest_directory
from hybrid_rag_mcp.stores.lexic import LexicalStore
from hybrid_rag_mcp.stores.vector import VectorStore


class StubEmbedder:
    dim = 4

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for i, _ in enumerate(texts):
            v = [0.0] * self.dim
            v[i % self.dim] = 1.0
            vectors.append(v)
        return vectors


def _write_docs(corpus: Path) -> None:
    (corpus / "a.txt").write_text("# Tema A\n" + "O cache usa Redis. " * 20, encoding="utf-8")
    (corpus / "b.txt").write_text("# Tema B\n" + "O barramento usa Kafka. " * 20, encoding="utf-8")


def test_store_persiste_e_restaura_chunks(tmp_path: Path) -> None:
    corpus, data = tmp_path / "corpus", tmp_path / "data"
    corpus.mkdir()
    _write_docs(corpus)
    settings = Settings(qdrant_path=str(data / "qdrant"), corpus_dir=str(corpus))

    vector = VectorStore(settings, StubEmbedder())
    lexical = LexicalStore()
    stats = ingest_directory(str(corpus), settings, vector, lexical, StubEmbedder())
    n = stats["chunks"]
    assert n > 0
    assert len(vector.all_chunks()) == n
    assert len(lexical._chunks) == n
    vector.close()

    restored = VectorStore(settings, StubEmbedder())
    chunks = restored.all_chunks()
    assert len(chunks) == n

    lexical2 = LexicalStore()
    lexical2.upsert_chunks(chunks)
    hits = hybrid_search(restored, lexical2, "cache redis", top_k=2, bm25_top_k=10)
    assert hits
    assert hits[0].doc_name in {"a.txt", "b.txt"}
