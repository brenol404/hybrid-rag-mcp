from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from uuid import UUID

import qdrant_client
from qdrant_client.http import models as qm

from ..config import Settings
from ..models import DocumentChunk, SearchHit
from ..providers import EmbeddingProvider


class VectorStore:
    """Busca semântica sobre Qdrant em modo local (sem servidor Docker)."""

    COLLECTION = "chunks"

    def __init__(self, settings: Settings, embedder: EmbeddingProvider) -> None:
        if settings.qdrant_url:
            # Modo servidor: índice compartilhado, vários processos/sessões
            # simultâneas (ver docker-compose.yml). Sem lock de arquivo.
            self._client = qdrant_client.QdrantClient(url=settings.qdrant_url)
        else:
            # Modo embarcado (default): sem Docker, um processo por vez.
            path = Path(settings.qdrant_path)
            path.mkdir(parents=True, exist_ok=True)
            self._client = qdrant_client.QdrantClient(path=str(path))
        self._embedder = embedder
        # Nada consulta o embedder (nem o Ollama) aqui: a coleção e a dim só são
        # resolvidas no 1º uso (search/ingest). Construir o engine é barato.
        self._collection_lock = threading.Lock()

    @property
    def _dim(self) -> int:
        return self._embedder.dim

    def _ensure_collection(self) -> None:
        with self._collection_lock:
            if not self._client.collection_exists(self.COLLECTION):
                self._client.create_collection(
                    collection_name=self.COLLECTION,
                    vectors_config=qm.VectorParams(size=self._dim, distance=qm.Distance.COSINE),
                )

    def _scroll_all_points(self) -> list[qm.Record]:
        """Lê todos os pontos da coleção com scroll paginado (Batch > página única)."""
        return _scroll_all(self._client, self.COLLECTION)

    def upsert_chunks(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            return
        self._ensure_collection()
        vectors = self._embedder.embed([c.content for c in chunks])
        self._client.upsert(
            collection_name=self.COLLECTION,
            points=[
                qm.PointStruct(
                    id=_hash_point(c.chunk_id),
                    vector=v,
                    payload={
                        "chunk_id": c.chunk_id,
                        "doc": c.doc_name,
                        "text": c.content,
                        "hid": _hash_content(c.content),
                    },
                )
                for c, v in zip(chunks, vectors, strict=True)
            ],
        )

    def sync_chunks(self, chunks: list[DocumentChunk]) -> dict[str, int]:
        """Sincroniza o índice com a lista de chunks: adiciona os novos, remove órfãos.

        Idempotente por hash de conteúdo: re-rodar `ingest` sem mudanças resulta em
        `added=0` e `deleted=0` (sem re-embedding). Chunks cujo texto mudou ganham
        hash novo e os antigos são podados.
        """
        added = unchanged = deleted = 0
        if chunks:
            self._ensure_collection()
            target = {_hash_content(c.content) for c in chunks}
            stored = self._scroll_all_points()
            present: set[int] = set()
            stale_ids: list[int | str | UUID] = []
            for p in stored:
                if not p.payload:
                    continue
                hid = int(p.payload.get("hid") or _hash_content(p.payload.get("text", "")))
                present.add(hid)
                if hid not in target:
                    stale_ids.append(p.id)
            if stale_ids:
                self._client.delete(
                    collection_name=self.COLLECTION,
                    points_selector=qm.PointIdsList(points=stale_ids),
                )
            to_add = [c for c in chunks if _hash_content(c.content) not in present]
            added = len(to_add)
            unchanged = len(chunks) - added
            deleted = len(stale_ids)
            if to_add:
                self.upsert_chunks(to_add)
        return {"added": added, "deleted": deleted, "unchanged": unchanged}

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        self._ensure_collection()
        query_vector = self._embedder.embed([query])[0]
        hits = self._client.query_points(
            collection_name=self.COLLECTION,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        ).points
        out: list[SearchHit] = []
        for h in hits:
            if not h.payload:
                continue
            out.append(
                SearchHit(
                    chunk_id=h.payload["chunk_id"],
                    doc_name=h.payload["doc"],
                    content=h.payload["text"],
                    score=h.score,
                    strategy="vector",
                )
            )
        return out

    def all_chunks(self) -> list[DocumentChunk]:
        """Recarrega todos os chunks persistidos (base da persistência do BM25)."""
        if not self._client.collection_exists(self.COLLECTION):
            return []  # nada indexado ainda — não cria a coleção nem consulta embedder
        points = self._scroll_all_points()
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


def _hash_content(text: str) -> int:
    """Fingerprint do conteúdo normalizado — identifica o chunk ideal, não a posição."""
    normalized = " ".join(text.split())
    return int.from_bytes(
        hashlib.blake2b(normalized.encode("utf-8"), digest_size=16).digest(), "big"
    )


_SCROLL_BATCH = 1000


def _scroll_all(client: qdrant_client.QdrantClient, collection_name: str) -> list[qm.Record]:
    """Scroll paginado: itera todas as páginas da coleção (offset até None).

    `scroll` sem offset retorna apenas a primeira página (limit fixo); acima do
    batch os chunks simplesmente sumiam do sync/persistência. Aqui seguimos o
    `next_page_offset` até esgotar.
    """
    points: list[qm.Record] = []
    offset = None
    while True:
        page, offset = client.scroll(
            collection_name=collection_name,
            limit=_SCROLL_BATCH,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points.extend(page)
        if offset is None:
            return points
