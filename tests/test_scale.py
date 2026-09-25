"""Testes do bench de escala — matemática e construção do corpus, sem Ollama."""

from __future__ import annotations

from hybrid_rag_mcp.bench import build_scale_corpus
from hybrid_rag_mcp.metrics import percentiles


def test_percentiles_nearest_rank() -> None:
    xs = [float(i) for i in range(1, 101)]  # 1..100
    p = percentiles(xs)
    assert p["n"] == 100
    assert p["mean"] == 50.5
    assert p["p50"] == 50.0
    assert p["p95"] == 95.0
    assert p["p99"] == 99.0


def test_percentiles_vazio() -> None:
    assert percentiles([]) == {"n": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}


def test_build_scale_corpus_atinge_alvo_com_ids_unicos() -> None:
    base = [("a.txt", "O servidor escuta na porta 8443. " * 30)]
    chunks = build_scale_corpus(base, target_chunks=50, chunk_size=128, overlap=16)
    assert len(chunks) >= 50
    ids = [c.chunk_id for c in chunks]
    assert len(set(ids)) == len(ids)  # sem colisão de ponto no Qdrant
    docs = {c.doc_name for c in chunks}
    assert len(docs) > 1  # mais de uma cópia
    salted = {c.doc_name for c in chunks if "escala-" in c.content}
    assert salted == docs  # toda cópia tem ao menos o chunk de cauda com salt
