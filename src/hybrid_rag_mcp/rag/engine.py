from __future__ import annotations

import json
import time
from pathlib import Path

from ..config import Settings, get_settings
from ..models import AskResult, SearchHit, TraceStep
from ..providers import FallbackLLM, resolve_embedder
from ..providers.rerank import build_reranker
from ..stores import LexicalStore, VectorStore
from .hybrid import hybrid_search
from .ingestion import ingest_directory

SYSTEM_PROMPT = (
    "Você é um assistente técnico que responde usando APENAS o contexto fornecido. "
    "Se a resposta não estiver no contexto, diga explicitamente que não encontrou. "
    "Cite a fonte de cada afirmação entre colchetes, ex.: [manual.pdf]."
)


class RAGEngine:
    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        self._settings = settings
        self._embedder = resolve_embedder(settings)
        self._vector = VectorStore(settings, self._embedder)
        self._lexical = LexicalStore()
        self._reranker = build_reranker(settings)
        self._llm = FallbackLLM(settings)
        self._restore_lexical()

    def _restore_lexical(self) -> None:
        """Persistência do BM25: reconstrói o índice léxico a partir dos chunks salvos no Qdrant."""
        stored = self._vector.all_chunks()
        if stored:
            self._lexical.upsert_chunks(stored)

    def close(self) -> None:
        """Libera o lock do Qdrant local. Use entre instâncias no mesmo processo."""
        self._vector.close()

    # ----- Ingestão ----------------------------------------------------------
    def ingest(self, corpus_dir: str | None = None) -> dict[str, int]:
        dir_to_scan = corpus_dir or self._settings.corpus_dir
        return ingest_directory(
            dir_to_scan,
            self._settings,
            self._vector,
            self._lexical,
            self._embedder,
        )

    # ----- Busca --------------------------------------------------------------
    def search(self, query: str, top_k: int | None = None) -> list[SearchHit]:
        top_k = top_k or self._settings.top_k
        return hybrid_search(
            self._vector,
            self._lexical,
            query,
            top_k=top_k,
            bm25_top_k=self._settings.bm25_top_k,
            reranker=self._reranker,
            rerank_budget=self._settings.rerank_budget,
        )

    # ----- Geração com rastreabilidade ----------------------------------------
    def ask(self, question: str, top_k: int | None = None) -> AskResult:
        settings = self._settings
        top_k = top_k or settings.top_k
        trace: list[TraceStep] = [TraceStep("ask", question)]
        t0 = time.perf_counter()

        sources = hybrid_search(
            self._vector,
            self._lexical,
            question,
            top_k=top_k,
            bm25_top_k=settings.bm25_top_k,
            reranker=self._reranker,
            rerank_budget=settings.rerank_budget,
        )
        trace.append(TraceStep("retrieval", f"{len(sources)} fontes (hybrid RRF)"))

        context = "\n\n".join(f"[{s.doc_name}] {s.content}" for s in sources)
        try:
            resp = self._llm.complete(
                SYSTEM_PROMPT, f"Contexto:\n{context}\n\nPergunta: {question}"
            )
        except RuntimeError as exc:
            trace.append(TraceStep("generation", str(exc), ok=False))
            raise
        trace.append(
            TraceStep(
                "generation",
                f"provedor={resp.provider} modelo={resp.model} em {time.perf_counter() - t0:.2f}s",
            )
        )
        self._append_audit(question, sources, resp.provider, resp.model, time.perf_counter() - t0)

        return AskResult(
            answer=resp.text, sources=sources, provider=resp.provider, model=resp.model, trace=trace
        )

    # ----- Auditoria ------------------------------------------------------------
    def _append_audit(
        self, question: str, sources: list[SearchHit], provider: str, model: str, elapsed: float
    ) -> None:
        path = Path(self._settings.audit_log)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "question": question,
            "provider": provider,
            "model": model,
            "elapsed_s": round(elapsed, 3),
            "sources": [{"doc": s.doc_name, "score": round(s.score, 4)} for s in sources],
        }
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
