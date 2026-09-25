"""Cache semântico de respostas: evita regerar quando a pergunta já foi feita.

Estratégia em duas camadas, persistida em JSONL (`data/cache.jsonl`):
  1. Normalização exata — perguntas idênticas (ignorando caixa/whitespace) batem;
  2. Semântica — a pergunta é embedded e comparada por cosseno com as entradas;
     acima de `cache_sim_threshold` é devolvido o hit (TTL via `cache_ttl_sec`).
Ao economizar uma geração inteira, o cache é o maior "otimizador de tokens".

Escrita é append-log (uma linha por store, sem reescrever o arquivo inteiro);
a cada `_COMPACT_EVERY` gravações o arquivo é reescrito a partir da memória
(deduplicada e limitada a `cache_max_entries`) para não crescer sem limite.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import Settings
from ..models import SearchHit
from ..providers import EmbeddingProvider


@dataclass(frozen=True)
class CacheHit:
    answer: str
    provider: str
    model: str
    sources: list[SearchHit]
    similarity: float


class SemanticCache:
    _COMPACT_EVERY = 64

    def __init__(self, settings: Settings, embedder: EmbeddingProvider) -> None:
        self._path = Path(settings.cache_path)
        self._embedder = embedder
        self._threshold = settings.cache_sim_threshold
        self._ttl = settings.cache_ttl_sec
        self._max_entries = settings.cache_max_entries
        self._lock = threading.Lock()
        self._writes_since_compact = 0
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._entries = self._load()

    # ---- Persistência -----------------------------------------------------
    def _load(self) -> list[dict[str, Any]]:
        entries: dict[str, dict[str, Any]] = {}
        if not self._path.exists():
            return []
        now = time.time()
        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if now - entry["ts"] <= self._ttl:
                    # append-log pode ter duplicatas da mesma pergunta: vale a mais recente
                    entries[entry["q"]] = entry
        # Mesmo teto da memória: entre compactações o arquivo pode ficar maior,
        # mas a carga só mantém as `max_entries` mais recentes.
        return list(entries.values())[-self._max_entries :]

    def _flush(self) -> None:
        """Reescreve o arquivo inteiro a partir da memória (compactação)."""
        with self._path.open("w", encoding="utf-8") as fh:
            for entry in self._entries:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _append_entry(self, entry: dict[str, Any]) -> None:
        """Append-log: uma linha nova sem reescrever o arquivo (O(1) de I/O)."""
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ---- Interface ----------------------------------------------------------
    @staticmethod
    def _normalize(question: str) -> str:
        return " ".join(question.lower().split())

    def lookup(self, question: str) -> CacheHit | None:
        q = self._normalize(question)
        now = time.time()
        with self._lock:
            for entry in self._entries:
                if now - entry["ts"] > self._ttl:
                    continue
                if entry["q"] == q:
                    return self._to_hit(entry, similarity=1.0)
            query_vec = self._embedder.embed([q])[0]
            best: dict[str, Any] | None = None
            best_sim = 0.0
            for entry in self._entries:
                if now - entry["ts"] > self._ttl:
                    continue
                cos = _cosine(query_vec, entry["emb"])
                if cos > best_sim:
                    best_sim = cos
                    best = entry
            if best is not None and best_sim >= self._threshold:
                return self._to_hit(best, similarity=best_sim)
        return None

    def store(
        self,
        question: str,
        answer: str,
        provider: str,
        model: str,
        sources: list[SearchHit],
    ) -> None:
        q = self._normalize(question)
        entry: dict[str, Any] = {
            "q": q,
            "emb": self._embedder.embed([q])[0],
            "answer": answer,
            "provider": provider,
            "model": model,
            "sources": [
                {"doc": s.doc_name, "score": s.score, "content": s.content[:400]} for s in sources
            ],
            "ts": time.time(),
        }
        with self._lock:
            self._entries = [e for e in self._entries if e["q"] != q]
            self._entries.append(entry)
            if len(self._entries) > self._max_entries:
                self._entries = sorted(self._entries, key=lambda e: e["ts"])
                self._entries = self._entries[-self._max_entries :]
            self._writes_since_compact += 1
            if self._writes_since_compact >= self._COMPACT_EVERY:
                self._flush()
                self._writes_since_compact = 0
            else:
                self._append_entry(entry)

    def clear(self) -> None:
        with self._lock:
            self._entries = []
            self._writes_since_compact = 0
            self._flush()

    @property
    def size(self) -> int:
        return len(self._entries)

    # ---- Helpers --------------------------------------------------------------
    @staticmethod
    def _to_hit(entry: dict[str, Any], similarity: float) -> CacheHit:
        sources = [
            SearchHit(
                chunk_id=f"cache-{i}",
                doc_name=s["doc"],
                content=s["content"],
                score=float(s["score"]),
                strategy="cache",
            )
            for i, s in enumerate(entry["sources"])
        ]
        return CacheHit(
            answer=entry["answer"],
            provider=entry["provider"],
            model=entry["model"],
            sources=sources,
            similarity=similarity,
        )


def _cosine(a: list[float], b: list[float]) -> float:
    dot: float = sum(x * y for x, y in zip(a, b, strict=True))
    na: float = sum(x * x for x in a) ** 0.5
    nb: float = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
