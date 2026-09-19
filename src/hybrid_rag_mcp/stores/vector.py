from __future__ import annotations

import hashlib
from pathlib import Path

import qdrant_client
from qdrant_client.http import models as qm

from ..config import Settings
from ..models import DocumentChunk, SearchHit
from ..providers import EmbeddingProvider


class VectorStore:
    """Busca semântica sobre Qdrant em modo local (sem servidor Docker)."""

    COLLECTION = "chunks"

    def __init__(self, settings: Settings, embedder: EmbeddingProvider) -> None:
        path = Path(settings.qdrant_path)
        path.mkdir(parents=True, exist_ok=True)
        self._client = qdrant_client.QdrantClient(path=str(path))
        self._embedder = embedder
        self._dim = embedder.dim
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if not self._client.collection_exists(self.COLLECTION):
            self._client.create_collection(
                collection_name=self.COLLECTION,
                vectors_config=qm.VectorParams(size=self._dim, distance=qm.Distance.COSINE),
            )

    def upsert_chunks(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            return
        vectors = self._embedder.embed([c.content for c in chunks])
        self._client.upsert(
            collection_name=self.COLLECTION,
            points=[
                qm.PointStruct(
                    id=_hash_point(c.chunk_id),
                    vector=v,
                    payload={"chunk_id": c.chunk_id, "doc": c.doc_name, "text": c.content},
                )
                for c, v in zip(chunks, vectors, strict=True)
            ],
        )

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        query_vector = self._embedder.embed([query])[0]
        hits = self._client.query_points(
            collection_name=self.COLLECTION,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        ).points
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

    def all_chunks(self) -> list[DocumentChunk]:
        """Recarrega todos os chunks persistidos (base da persistência do BM25)."""
        points = self._client.scroll(
            collection_name=self.COLLECTION,
            limit=10000,
            with_payload=True,
            with_vectors=False,
        )[0]
        return [
            DocumentChunk(
                chunk_id=p.payload["chunk_id"],
                doc_name=p.payload["doc"],
                content=p.payload["text"],
                index=0,
            )
            for p in points
            if p.payload
        ]

    def close(self) -> None:
        self._client.close()


def _hash_point(chunk_id: str) -> int:
    return int.from_bytes(hashlib.blake2b(chunk_id.encode("utf-8"), digest_size=16).digest(), "big")
