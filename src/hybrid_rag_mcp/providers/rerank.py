from __future__ import annotations

import httpx

from ..config import Settings


class Reranker:
    """Cross-encoder via endpoint /api/rerank do Ollama (versão >= 0.36, ex.: bge-reranker-v2-m3).

    Se o modelo não estiver configurado ou o endpoint falhar, a chamada
    simplesmente não re-ordena — degradação graciosa, nunca quebra o pipeline.
    """

    def __init__(self, settings: Settings) -> None:
        self._model = settings.rerank_model
        self._url = f"{settings.ollama_host.rstrip('/')}/api/rerank"
        self._http = httpx.Client(timeout=60)

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not self._model or not texts:
            return [0.0] * len(texts)
        try:
            resp = self._http.post(
                self._url,
                json={"model": self._model, "query": query, "documents": texts},
            )
            if resp.status_code >= 400:
                return [0.0] * len(texts)
            results = resp.json().get("results", [])
        except Exception:  # noqa: BLE001 - endpoint ausente/offline: segue sem re-ranking
            return [0.0] * len(texts)
        ranked = [(r["index"], r.get("relevance_score", 0.0)) for r in results]
        scores = [0.0] * len(texts)
        for index, score in ranked:
            if 0 <= index < len(texts):
                scores[index] = score
        return scores


def build_reranker(settings: Settings) -> Reranker | None:
    return Reranker(settings) if settings.rerank_model else None
