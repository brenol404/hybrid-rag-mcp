"""Testes de persistência — Qdrant + BM25 sem necessidade de Ollama (embedder stub)."""

from __future__ import annotations

from pathlib import Path

from hybrid_rag_mcp.config import Settings
from hybrid_rag_mcp.rag import engine as engine_module
from hybrid_rag_mcp.rag.engine import RAGEngine
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


def test_ingest_e_idempotente(tmp_path: Path) -> None:
    corpus, data = tmp_path / "corpus", tmp_path / "data"
    corpus.mkdir()
    _write_docs(corpus)
    settings = Settings(qdrant_path=str(data / "qdrant"), corpus_dir=str(corpus))

    vector = VectorStore(settings, StubEmbedder())
    lexical = LexicalStore()
    first = ingest_directory(str(corpus), settings, vector, lexical, StubEmbedder())
    assert first["added"] > 0 and first["deleted"] == 0

    second = ingest_directory(str(corpus), settings, vector, lexical, StubEmbedder())
    assert second["added"] == 0
    assert second["deleted"] == 0
    assert second["unchanged"] == second["chunks"]
    assert len(vector.all_chunks()) == first["chunks"]
    assert len(lexical._chunks) == first["indexed"]
    vector.close()


def test_ingest_incremental_embeds_so_o_que_mudou(tmp_path: Path, monkeypatch) -> None:
    corpus, data = tmp_path / "corpus", tmp_path / "data"
    corpus.mkdir()
    _write_docs(corpus)
    settings = Settings(qdrant_path=str(data / "qdrant"), corpus_dir=str(corpus))

    embedded: list[str] = []
    counted = StubEmbedder()
    original = counted.embed
    counted.embed = lambda texts: (embedded.extend(texts), original(texts))[1]  # type: ignore[method-assign]

    vector = VectorStore(settings, counted)
    lexical = LexicalStore()
    ingest_directory(str(corpus), settings, vector, lexical, counted)
    first_embeds = len(embedded)

    (corpus / "a.txt").write_text(
        "# Tema A alterado\n" + "O cache agora usa Valkey. " * 20, encoding="utf-8"
    )
    stats = ingest_directory(str(corpus), settings, vector, lexical, counted)
    assert stats["added"] > 0
    assert stats["deleted"] > 0
    assert len(embedded) == first_embeds + stats["added"]
    assert len(lexical._chunks) == stats["indexed"]
    hits = hybrid_search(vector, lexical, "valkey", top_k=2, bm25_top_k=10)
    assert hits and hits[0].doc_name == "a.txt"
    vector.close()


def test_ingest_remove_doc_poda_orfao(tmp_path: Path) -> None:
    corpus, data = tmp_path / "corpus", tmp_path / "data"
    corpus.mkdir()
    _write_docs(corpus)
    settings = Settings(qdrant_path=str(data / "qdrant"), corpus_dir=str(corpus))

    vector = VectorStore(settings, StubEmbedder())
    lexical = LexicalStore()
    ingest_directory(str(corpus), settings, vector, lexical, StubEmbedder())
    before = len(vector.all_chunks())

    (corpus / "b.txt").unlink()
    stats = ingest_directory(str(corpus), settings, vector, lexical, StubEmbedder())
    assert stats["added"] == 0
    assert stats["deleted"] > 0
    assert len(vector.all_chunks()) == before - stats["deleted"]
    staying = {c.doc_name for c in vector.all_chunks()}
    assert "b.txt" not in staying
    assert "a.txt" in staying
    vector.close()


def test_scroll_pagina_acima_do_batch(tmp_path: Path) -> None:
    """Corpora maiores que o batch de scroll (1000) não perdem chunks.

    O `scroll` sem offset devolve só a primeira página; o helper `_scroll_all`
    segue o `next_page_offset` até esgotar. Aqui forçamos >1 página e re-rodamos
    o sync para provar que nada some.
    """
    corpus, data = tmp_path / "corpus", tmp_path / "data"
    corpus.mkdir()
    for k in range(18):
        sentences = " ".join(f"frase {i} do tema {k} com token unico {k}-{i}." for i in range(800))
        (corpus / f"doc-{k}.txt").write_text(f"# Tema {k}\n{sentences}", encoding="utf-8")
    settings = Settings(qdrant_path=str(data / "qdrant"), corpus_dir=str(corpus))
    vector = VectorStore(settings, StubEmbedder())
    lexical = LexicalStore()
    stats = ingest_directory(str(corpus), settings, vector, lexical, StubEmbedder())
    n = stats["chunks"]
    assert n > 1000, f"esperado corpus > 1000 chunks para paginar, veio {n}"

    assert len(vector.all_chunks()) == n

    again = ingest_directory(str(corpus), settings, vector, lexical, StubEmbedder())
    assert again["added"] == 0
    assert again["deleted"] == 0
    assert again["unchanged"] == n
    assert len(vector.all_chunks()) == n
    vector.close()


def test_engine_warmup_restaura_lexico_para_uso_direto(tmp_path: Path, monkeypatch) -> None:
    """Contrato do init lazy: quem usa `rag._lexical` direto precisa de `warmup()`.

    Regressão real: após a restauração do BM25 virar preguiçosa,
    `tools/grid_search.py` lia o léxico vazio (0 chunks) e rodava o tuning
    só no vetorial, em silêncio. Sem Ollama (embedder stub).
    """
    monkeypatch.setattr(engine_module, "resolve_embedder", lambda settings: StubEmbedder())
    corpus, data = tmp_path / "corpus", tmp_path / "data"
    corpus.mkdir()
    _write_docs(corpus)
    settings = Settings(
        qdrant_path=str(data / "qdrant"),
        corpus_dir=str(corpus),
        cache_path=str(data / "cache.jsonl"),
        audit_log=str(data / "audit.jsonl"),
    )

    first = RAGEngine(settings)
    first.ingest(str(corpus))
    first.close()

    fresh = RAGEngine(settings)
    assert len(fresh._lexical._chunks) == 0  # lazy: nada restaurado ainda
    fresh.warmup()
    assert len(fresh._lexical._chunks) > 0
    hits = hybrid_search(fresh._vector, fresh._lexical, "cache redis", top_k=2, bm25_top_k=10)
    assert hits and hits[0].doc_name == "a.txt"
    fresh.close()  # fecha Qdrant + HTTP: não deve levantar
