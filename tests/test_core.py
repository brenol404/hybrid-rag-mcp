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


def _brute_force(
    corpus: list[list[str]], query: list[str], k1: float = 1.5, b: float = 0.75
) -> list[float]:
    """BM25 ingênuo em loops aninhados — referência independente da implementação."""
    import math
    from collections import Counter

    n = len(corpus)
    if n == 0:
        return []
    tfs = [Counter(d) for d in corpus]
    dl = [sum(c.values()) for c in tfs]
    avgdl = sum(dl) / n
    if avgdl == 0:
        return [0.0] * n
    df: Counter[str] = Counter(t for d in corpus for t in set(d))
    scores = [0.0] * n
    for q in set(query):
        if q not in df:
            continue
        idf = math.log(1.0 + (n - df[q] + 0.5) / (df[q] + 0.5))
        for i, tf in enumerate(tfs):
            f = tf.get(q, 0)
            if f:
                scores[i] += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl[i] / avgdl))
    return scores


def test_bm25_vetorizado_igual_forca_bruta() -> None:
    import random

    from hybrid_rag_mcp.stores.lexic import BM25

    rng = random.Random(42)
    vocab = [f"t{i}" for i in range(60)]
    for trial in range(5):
        corpus = [[rng.choice(vocab) for _ in range(rng.randint(0, 40))] for _ in range(300)]
        query = [rng.choice(vocab) for _ in range(8)]
        model = BM25()
        model.fit(corpus)
        got, want = model.score_all(query), _brute_force(corpus, query)
        assert len(got) == len(want) == 300
        assert max(abs(g - w) for g, w in zip(got, want, strict=True)) < 1e-9


def test_bm25_edges() -> None:
    from hybrid_rag_mcp.stores.lexic import BM25

    assert BM25().score_all(["x"]) == []
    m = BM25()
    m.fit([["a", "b"], ["b", "c"]])
    assert m.score_all(["zzz"]) == [0.0, 0.0]
    assert m.score_all([]) == [0.0, 0.0]
    empty = BM25()
    empty.fit([[], []])
    assert empty.score_all(["a"]) == [0.0, 0.0]
    assert m.score_all(["b", "b"]) == m.score_all(["b"])  # sets: repetição não pesa
