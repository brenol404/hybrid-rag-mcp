from __future__ import annotations

from ..models import SearchHit
from ..providers.rerank import Reranker
from ..stores import LexicalStore, VectorStore


def rrf_fusion(
    vector_hits: list[SearchHit], lexical_hits: list[SearchHit], k: int = 60
) -> list[SearchHit]:
    """Combina resultados de busca vetorial e léxica via Reciprocal Rank Fusion (RRF)."""
    scores: dict[str, float] = {}
    merged: dict[str, SearchHit] = {}

    for rank, hit in enumerate(vector_hits, start=1):
        scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
        merged[hit.chunk_id] = hit

    for rank, hit in enumerate(lexical_hits, start=1):
        scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
        merged[hit.chunk_id] = hit

    ranked = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [
        SearchHit(
            chunk_id=cid,
            doc_name=merged[cid].doc_name,
            content=merged[cid].content,
            score=scores[cid],
            strategy="hybrid",
        )
        for cid in ranked
    ]


def hybrid_search(
    vector_store: VectorStore,
    lexical_store: LexicalStore,
    query: str,
    top_k: int,
    bm25_top_k: int,
    reranker: Reranker | None = None,
    rerank_budget: int = 20,
) -> list[SearchHit]:
    vector_hits = vector_store.search(query, top_k=top_k)
    lexical_hits = lexical_store.search(query, top_k=bm25_top_k)
    fused = rrf_fusion(vector_hits, lexical_hits, k=60)

    if reranker is not None and rerank_budget > 0:
        budget = fused[:rerank_budget]
        texts = [h.content for h in budget]
        scores = reranker.score(query, texts)
        budget = [
            SearchHit(
                chunk_id=h.chunk_id,
                doc_name=h.doc_name,
                content=h.content,
                score=float(score),
                strategy="hybrid",
            )
            for h, score in sorted(
                zip(budget, scores, strict=True), key=lambda t: t[1], reverse=True
            )
        ]
        return budget[:top_k]

    return fused[:top_k]
