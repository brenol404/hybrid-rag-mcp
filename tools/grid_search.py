"""Grid-search dos hiperparâmetros da fusão híbrida (RRF) para maximizar recall@1.

Reutiliza o índice já persistido em `data/` (não re-embeda o corpus) e cacheia os
embeddings das queries. Assim o tuning roda em segundos, sobre as mesmas 36 consultas
do eval.

Uso:
    python tools/grid_search.py                       # varre o grid cheio
    python tools/grid_search.py --json eval/grid_results.json
    python tools/grid_search.py --top 5               # mostra só o top-5

Resultado sugerido volta num formato pronto pra fixar nos defaults do config.py.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from hybrid_rag_mcp.config import get_settings
from hybrid_rag_mcp.eval import evaluate, load_datasets
from hybrid_rag_mcp.rag.engine import RAGEngine
from hybrid_rag_mcp.rag.hybrid import hybrid_search

KS = (1, 3, 5)


class _CachedVector:
    """Wrapper do VectorStore com cache de embeddings por query."""

    def __init__(self, store) -> None:
        self._store = store
        self._embed = store._embedder
        self._cache: dict[str, list[float]] = {}

    def search(self, query: str, top_k: int):
        if query not in self._cache:
            self._cache[query] = self._embed.embed([query])[0]
        hits = self._store._client.query_points(
            collection_name=self._store.COLLECTION,
            query=self._cache[query],
            limit=top_k,
            with_payload=True,
        ).points
        from hybrid_rag_mcp.models import SearchHit

        return [
            SearchHit(
                chunk_id=h.payload["chunk_id"],
                doc_name=h.payload["doc"],
                content=h.payload["text"],
                score=h.score,
                strategy="vector",
            )
            for h in hits
        ]


def _engine_for(cached, lexical, bm25_top_k: int, rrf_k: int, weights: tuple[float, float]):
    class _Engine:
        def search(self, query, top_k):
            return hybrid_search(
                cached,
                lexical,
                query,
                top_k=top_k,
                bm25_top_k=bm25_top_k,
                rrf_k=rrf_k,
                rrf_weights=weights,
            )

        def ingest(self, *args, **kwargs):
            return {}

    return _Engine()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default=None, help="grava os resultados como JSON")
    parser.add_argument("--top", type=int, default=10, help="linhas exibidas")
    args = parser.parse_args()

    rag = RAGEngine(get_settings())
    rag.warmup()  # restaura o BM25 persistido: os stores abaixo são usados direto
    cached = _CachedVector(rag._vector)
    dataset = load_datasets("eval/dataset.jsonl", "eval/dataset.real.jsonl")

    k_vals = [30, 60, 90]
    bm25_vals = [10, 20, 50]
    weights_vals = [(1.0, 1.0), (1.5, 1.0), (1.0, 1.5)]
    grid = list(itertools.product(k_vals, bm25_vals, weights_vals))

    rows: list[dict] = []
    for rrf_k, bm25_top_k, weights in grid:
        report = evaluate(
            _engine_for(cached, rag._lexical, bm25_top_k, rrf_k, weights),
            dataset,
            KS,
        )
        r1 = report.mean_recall(1)
        r3 = report.mean_recall(3)
        r5 = report.mean_recall(5)
        rows.append(
            {
                "rrf_k": rrf_k,
                "bm25_top_k": bm25_top_k,
                "w_vec": weights[0],
                "w_lex": weights[1],
                "recall@1": round(r1, 3),
                "recall@3": round(r3, 3),
                "recall@5": round(r5, 3),
            }
        )

    rows.sort(key=lambda r: r["recall@1"], reverse=True)
    best = rows[0]
    print(f"Grid: {len(rows)} combos · {len(dataset)} queries\n")
    print(f"{'rrf_k':>6} {'bm25':>5} {'w_vec':>6} {'w_lex':>6} {'r@1':>6} {'r@3':>6} {'r@5':>6}")
    for r in rows[: args.top]:
        print(
            f"{r['rrf_k']:>6} {r['bm25_top_k']:>5} {r['w_vec']:>6} {r['w_lex']:>6} "
            f"{r['recall@1']:>6} {r['recall@3']:>6} {r['recall@5']:>6}"
        )
    print(
        f"\nMelhor combo: rrf_k={best['rrf_k']}, bm25_top_k={best['bm25_top_k']}, "
        f"weights=({best['w_vec']}, {best['w_lex']}) → recall@1 = {best['recall@1']}"
    )

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"Salvo em {args.json}")
    rag.close()


if __name__ == "__main__":
    main()
