"""Bench de escala: latência e throughput do retrieval com N chunks (default 100k).

Metodologia honesta, leia antes de citar os números:
  - RECALL não é medido aqui (continua no eval curado, corpus real).
  - O corpus é SINTÉTICO por replicação: os 11 docs reais de examples/corpus
    são copiados com salt único por cópia (nomes e conteúdo distintos — o
    dedupe e o hash de ponto exigiriam isso). Serve para medir LATÊNCIA,
    THROUGHPUT e TAMANHO do índice, não qualidade.
  - As queries são as 36 reais do eval; o LLM fica de fora (ask não roda) —
    é bench do retrieval, que é o nosso código.
  - O índice vai para data/scale-bench/qdrant (NUNCA no data/qdrant real).

Uso:
    python tools/scale_bench.py --chunks 100000
    python tools/scale_bench.py --chunks 10000 --workers 4 --out /tmp/bench.json
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from hybrid_rag_mcp.bench import build_scale_corpus, percentiles
from hybrid_rag_mcp.config import get_settings
from hybrid_rag_mcp.eval import load_datasets
from hybrid_rag_mcp.rag.engine import RAGEngine
from hybrid_rag_mcp.rag.ingestion import SUPPORTED_EXTS, read_document


def _dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=int, default=100_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--qdrant-path", default="data/scale-bench/qdrant")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    settings = get_settings()
    base_docs: list[tuple[str, str]] = []
    for path in sorted(Path(settings.corpus_dir).rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
            base_docs.append((path.name, read_document(path)))
    print(f"Base: {len(base_docs)} docs reais; alvo: {args.chunks} chunks")

    t0 = time.perf_counter()
    chunks = build_scale_corpus(base_docs, args.chunks, settings.chunk_size, settings.chunk_overlap)
    build_s = time.perf_counter() - t0
    print(f"Corpus sintético: {len(chunks)} chunks em {build_s:.1f}s")

    with tempfile.TemporaryDirectory(prefix="scale-corpus-") as tmp:
        # Um .txt por documento escalado — reaproveita o ingest testado.
        by_doc: dict[str, list[str]] = {}
        for c in chunks:
            by_doc.setdefault(c.doc_name, []).append(c.content)
        for doc_name, parts in by_doc.items():
            Path(tmp, doc_name).write_text("\n".join(parts), encoding="utf-8")

        settings.qdrant_path = args.qdrant_path
        rag = RAGEngine(settings)
        t0 = time.perf_counter()
        stats = rag.ingest(tmp)
        ingest_s = time.perf_counter() - t0
        print(
            f"Ingest: {stats['indexed']} no índice em {ingest_s:.1f}s "
            f"({stats['indexed'] / ingest_s:.0f} chunks/s)"
        )

        queries = [
            q["query"] for q in load_datasets("eval/dataset.jsonl", "eval/dataset.real.jsonl")
        ]
        print(f"Queries: {len(queries)} (reais, do eval)")

        def timed_search(q: str) -> float:
            t = time.perf_counter()
            rag.search(q, top_k=5)
            return time.perf_counter() - t

        single = [timed_search(q) for q in queries]
        lat1 = percentiles([s * 1000 for s in single])
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            multi = list(pool.map(timed_search, queries * 3))
        latN = percentiles([s * 1000 for s in multi])

        disk = _dir_size_mb(Path(args.qdrant_path))
        rag.close()

    report = {
        "chunks": len(chunks),
        "ingest_s": round(ingest_s, 1),
        "ingest_chunks_per_s": round(len(chunks) / ingest_s, 1),
        "index_disk_mb": round(disk, 1),
        "latency_ms_single": {k: round(v, 1) for k, v in lat1.items()},
        "latency_ms_concurrent": {k: round(v, 1) for k, v in latN.items()},
        "workers": args.workers,
    }
    print("\n== Escala (retrieval, corpus sintético) ==")
    print(
        f"chunks={report['chunks']} ingest={report['ingest_s']}s "
        f"({report['ingest_chunks_per_s']}/s) disco={report['index_disk_mb']}MB"
    )
    print(f"latência 1 thread (ms): {report['latency_ms_single']}")
    print(f"latência {args.workers} threads (ms): {report['latency_ms_concurrent']}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Salvo em {args.out}")


if __name__ == "__main__":
    main()
