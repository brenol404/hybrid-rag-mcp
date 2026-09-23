"""Testes unitários — não precisam de Ollama nem de disco real (tmp_path)."""

from __future__ import annotations

import threading
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


def _mk_gen_chunks(gen: int, n: int) -> list[DocumentChunk]:
    return [
        DocumentChunk(f"c{gen}-{i}", f"doc-{gen}", f"chunk {i} da geração {gen}", i)
        for i in range(n)
    ]


def test_lexical_store_seguro_sob_rebuild_concorrente() -> None:
    """Rebuilds em paralelo com buscas nunca expõem mistura de versões.

    No código antigo, `rebuild` trocava `_chunks` antes de reencher o modelo:
    tamanhos alternados (60 -> 5) faziam leitores pegarem chunks novos (5) com
    modelo antigo (60) e estourarem IndexError. Com o snapshot imutável, leitores
    só veem pares (chunks, modelo) consistentes.
    """
    store = LexicalStore()
    store.rebuild(_mk_gen_chunks(gen=0, n=5))
    stop = threading.Event()
    errors: list[Exception] = []

    def searcher() -> None:
        while not stop.is_set():
            try:
                hits = store.search("chunk geração", top_k=3)
                assert all(h.doc_name.startswith("doc-") for h in hits)
                assert all(h.content.startswith("chunk ") for h in hits)
            except Exception as exc:  # noqa: BLE001 - registra qualquer falha e para
                errors.append(exc)
                stop.set()

    threads = [threading.Thread(target=searcher) for _ in range(6)]
    for t in threads:
        t.start()
    try:
        sizes = (5, 60, 5, 60, 5, 60)  # alternar tamanhos expõe mismatch de índice
        for gen in range(120):
            store.rebuild(_mk_gen_chunks(gen, sizes[gen % len(sizes)]))
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=10)
    assert not errors, errors[:3]
