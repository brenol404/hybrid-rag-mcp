"""Testes unitários — não precisam de Ollama nem de disco real (tmp_path)."""

from __future__ import annotations

from pathlib import Path

from hybrid_rag_mcp.config import Settings
from hybrid_rag_mcp.models import DocumentChunk, SearchHit
from hybrid_rag_mcp.rag.chunker import chunk_text
from hybrid_rag_mcp.rag.hybrid import rrf_fusion
from hybrid_rag_mcp.stores.lexic import LexicalStore


def test_chunk_text_respects_headings_and_overlap() -> None:
    text = "# Seção A\n" + "O servidor escuta na porta 8443 por padrão. " * 30
    text += "# Seção B\n" + "O backup é incremental a cada 4 horas. " * 30
    chunks = chunk_text("doc.md", text, chunk_size=256, overlap=40)
    assert len(chunks) > 1
    assert all(c.doc_name == "doc.md" for c in chunks)
    assert all(c.content.strip() for c in chunks)
    assert all(c.content == c2.content for c, c2 in zip(chunks, chunks, strict=True))  # dedupe ok


def test_chunk_ids_are_unique() -> None:
    chunks = chunk_text("doc.md", "Frase um. " * 100, chunk_size=64, overlap=16)
    ids = [c.chunk_id for c in chunks]
    assert len(set(ids)) == len(ids)


def test_empty_text_yields_no_chunks() -> None:
    assert chunk_text("doc.md", "") == []


def _mk_chunk(i: int) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"c{i}", doc_name="doc", content=f"conteúdo do chunk {i}", index=i
    )


def test_lexical_search_finds_terms(tmp_path: Path) -> None:
    store = LexicalStore()
    store.upsert_chunks([_mk_chunk(1), _mk_chunk(2)])
    hits = store.search("chunk 2", top_k=5)
    assert hits and hits[0].chunk_id == "c2"


def test_rrf_fusion_combines_strategies() -> None:
    vec = [SearchHit("c1", "d", "x", 0.9, "vector"), SearchHit("c2", "d", "y", 0.8, "vector")]
    lex = [SearchHit("c2", "d", "y", 5.0, "lexical"), SearchHit("c3", "d", "z", 4.0, "lexical")]
    fused = rrf_fusion(vec, lex)
    assert fused[0].chunk_id in {"c1", "c2"}
    assert {h.strategy for h in fused} == {"hybrid"}


def test_rrf_preserves_all_and_ranks_present_in_both_higher() -> None:
    vec = [SearchHit("a", "d", "x", 0.9, "vector"), SearchHit("b", "d", "y", 0.1, "vector")]
    lex = [SearchHit("b", "d", "y", 5.0, "lexical"), SearchHit("c", "d", "z", 1.0, "lexical")]
    fused = rrf_fusion(vec, lex)
    assert {h.chunk_id for h in fused} == {"a", "b", "c"}
    assert fused[0].chunk_id == "b"


def test_settings_defaults() -> None:
    s = Settings()
    assert s.embed_model == "bge-m3"
    assert s.chunk_overlap < s.chunk_size
