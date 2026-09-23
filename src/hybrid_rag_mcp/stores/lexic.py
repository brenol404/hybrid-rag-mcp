from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from ..models import DocumentChunk, SearchHit

_STOPWORDS = {
    "a",
    "as",
    "o",
    "os",
    "um",
    "uma",
    "uns",
    "umas",
    "de",
    "do",
    "da",
    "dos",
    "das",
    "em",
    "no",
    "na",
    "nos",
    "nas",
    "para",
    "por",
    "com",
    "e",
    "ou",
    "que",
    "se",
    "the",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "at",
    "for",
    "is",
    "are",
    "was",
    "it",
    "qual",
    "quais",
    "como",
    "quando",
    "onde",
    "quanto",
}


class BM25:
    """BM25 com idf suavizado (log(1 + ...)) — robusto em corpora pequenos."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._tfs: list[Counter[str]] = []
        self._dl: list[int] = []
        self._avgdl = 0.0
        self._df: Counter[str] = Counter()
        self._n = 0
        self._idf: dict[str, float] = {}

    def fit(self, corpus: list[list[str]]) -> None:
        self._n = len(corpus)
        self._tfs = [Counter(tokens) for tokens in corpus]
        self._dl = [sum(c.values()) for c in self._tfs]
        self._avgdl = sum(self._dl) / self._n if self._n else 0.0
        self._df = Counter(term for c in corpus for term in set(c))
        self._idf = {
            term: math.log(1.0 + (self._n - freq + 0.5) / (freq + 0.5))
            for term, freq in self._df.items()
        }

    def score_all(self, query: list[str]) -> list[float]:
        if self._n == 0:
            return []
        scores = [0.0] * self._n
        for qterm in set(query):
            idf = self._idf.get(qterm, 0.0)
            if idf == 0.0:
                continue
            for i, tf in enumerate(self._tfs):
                freq = tf.get(qterm, 0)
                if freq:
                    denom = freq + self._k1 * (1 - self._b + self._b * self._dl[i] / self._avgdl)
                    scores[i] += idf * (freq * (self._k1 + 1)) / denom
        return scores


@dataclass(frozen=True)
class _LexicalSnapshot:
    """Estado imutável do índice léxico: (chunks, modelo) sempre juntos.

    `rebuild` monta o novo modelo em variável local e troca o snapshot inteiro
    num único assignment — atômico sob GIL — de modo que nenhuma leitura
    concorrente vê uma mistura de versões (chunks novos + modelo antigo e
    vice-versa).
    """

    chunks: tuple[tuple[str, str, str], ...]
    model: BM25


class LexicalStore:
    """Busca léxica BM25 em memória, sem dependências externas.

    Thread-safe para leituras concorrentes: leitores seguram uma referência ao
    snapshot (`_snapshot`) — sempre consistente — enquanto `rebuild`/`upsert`
    preparam a nova versão fora do estado publicado.
    """

    def __init__(self) -> None:
        self._snapshot = _LexicalSnapshot((), BM25())

    @property
    def _chunks(self) -> list[tuple[str, str, str]]:
        """Compat com inspeção/tests: expõe o estado atual como lista."""
        return list(self._snapshot.chunks)

    def upsert_chunks(self, chunks: list[DocumentChunk]) -> None:
        snap = self._snapshot
        self.rebuild(
            [
                DocumentChunk(chunk_id=cid, doc_name=doc, content=text, index=0)
                for cid, doc, text in list(snap.chunks)
                + [(c.chunk_id, c.doc_name, c.content) for c in chunks]
            ]
        )

    def rebuild(self, chunks: list[DocumentChunk]) -> None:
        stored = tuple((c.chunk_id, c.doc_name, c.content) for c in chunks)
        model = BM25()
        model.fit([_tokenize(c.content) for c in chunks])
        self._snapshot = _LexicalSnapshot(stored, model)

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        snap = self._snapshot
        if not snap.chunks:
            return []
        scores = snap.model.score_all(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [
            SearchHit(
                chunk_id=snap.chunks[i][0],
                doc_name=snap.chunks[i][1],
                content=snap.chunks[i][2],
                score=float(scores[i]),
                strategy="lexical",
            )
            for i in ranked[:top_k]
        ]


def _tokenize(text: str) -> list[str]:
    words = (t.strip().lower() for t in text.replace(".", " ").replace(",", " ").split())
    return [w for w in words if w and w not in _STOPWORDS]
