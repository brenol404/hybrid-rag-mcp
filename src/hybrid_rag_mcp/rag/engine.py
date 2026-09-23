from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from pathlib import Path

from ..config import Settings, get_settings
from ..models import AskResult, SearchHit, TraceStep
from ..optimize import SemanticCache, make_compressor
from ..providers import FallbackLLM, resolve_embedder
from ..providers.rerank import build_reranker
from ..stores import LexicalStore, VectorStore
from .agent import run_agent
from .hybrid import hybrid_search
from .ingestion import ingest_directory


class RAGEngine:
    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        self._settings = settings
        self._embedder = resolve_embedder(settings)
        self._vector = VectorStore(settings, self._embedder)
        self._lexical = LexicalStore()
        self._reranker = build_reranker(settings)
        self._llm = FallbackLLM(settings)
        self._cache = SemanticCache(settings, self._embedder) if settings.cache_enabled else None
        self._compress = make_compressor(settings.context_compression)
        # Restauração do BM25 é preguiçosa (1º search/ask): construir o engine
        # não toca em Ollama nem no Qdrant.
        self._ready = False
        self._ready_lock = threading.Lock()

    def _ensure_ready(self) -> None:
        """Reconstrói o índice léxico a partir dos chunks persistidos, sob demanda."""
        if not self._ready:
            with self._ready_lock:
                if not self._ready:
                    self._restore_lexical()
                    self._ready = True

    def warmup(self) -> None:
        """Restaura o índice léxico agora (sem buscar nada).

        Necessário apenas para quem usa os stores internos (`_vector`,
        `_lexical`) diretamente, sem passar por `search`/`ask`/`ingest` —
        ex.: `tools/grid_search.py`. O caminho normal já restaura sozinho.
        """
        self._ensure_ready()

    def _restore_lexical(self) -> None:
        """Persistência do BM25: reconstrói o índice léxico a partir dos chunks salvos no Qdrant."""
        stored = self._vector.all_chunks()
        if stored:
            self._lexical.upsert_chunks(stored)

    def close(self) -> None:
        """Libera recursos: lock do Qdrant local + clientes HTTP. Use entre instâncias no mesmo processo."""
        self._vector.close()
        if self._reranker is not None:
            self._reranker.close()
        self._llm.close()

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
        self._ensure_ready()
        top_k = top_k or self._settings.top_k
        return hybrid_search(
            self._vector,
            self._lexical,
            query,
            top_k=top_k,
            bm25_top_k=self._settings.bm25_top_k,
            reranker=self._reranker,
            rerank_budget=self._settings.rerank_budget,
            rrf_k=self._settings.rrf_k,
            rrf_weights=(self._settings.rrf_w_vector, self._settings.rrf_w_lexical),
        )

    # ----- Geração com rastreabilidade (agente multi-step) --------------------
    def ask(
        self,
        question: str,
        top_k: int | None = None,
        on_event: Callable[[str], None] | None = None,
        on_tokens: Callable[[str], None] | None = None,
    ) -> AskResult:
        settings = self._settings
        top_k = top_k or settings.top_k
        t0 = time.perf_counter()

        cache_hit = None
        if self._cache is not None:
            cache_hit = self._cache.lookup(question)
        if cache_hit is not None:
            result = AskResult(
                answer=cache_hit.answer,
                sources=cache_hit.sources,
                provider=cache_hit.provider,
                model=cache_hit.model,
                trace=[
                    TraceStep(
                        "cache",
                        f"resposta do cache (similaridade={cache_hit.similarity:.3f})",
                    )
                ],
                iterations=0,
                cache_hit=True,
            )
            result.trace.append(
                TraceStep("done", f"cache em {time.perf_counter() - t0:.2f}s, geração economizada")
            )
            self._append_audit(
                question,
                result.sources,
                result.provider,
                result.model,
                elapsed=time.perf_counter() - t0,
                iterations=0,
                cached=True,
            )
            return result

        # Cache miss: agora sim restaura o índice léxico (1ª vez) e recupera.
        self._ensure_ready()

        def retrieve(query: str, k: int) -> list[SearchHit]:
            return hybrid_search(
                self._vector,
                self._lexical,
                query,
                top_k=k,
                bm25_top_k=settings.bm25_top_k,
                reranker=self._reranker,
                rerank_budget=settings.rerank_budget,
                rrf_k=settings.rrf_k,
                rrf_weights=(settings.rrf_w_vector, settings.rrf_w_lexical),
            )

        result = run_agent(
            question,
            retrieve=retrieve,
            generate=lambda system, user: self._llm.complete(system, user),
            generate_stream=lambda system, user: self._llm.complete_stream(system, user),
            max_iterations=settings.ask_max_iterations,
            top_k=top_k,
            on_event=on_event,
            on_tokens=on_tokens,
            compress=self._compress,
        )
        result.trace.append(
            TraceStep(
                "done",
                f"concluído em {time.perf_counter() - t0:.2f}s, {result.iterations} iterações",
            )
        )
        if self._compress is not None:
            result.trace.append(
                TraceStep(
                    "compression",
                    f"contexto comprimido (nível {settings.context_compression}) "
                    "antes do prompt do LLM",
                )
            )
        if self._cache is not None:
            self._cache.store(
                question, result.answer, result.provider, result.model, result.sources
            )
        self._append_audit(
            question,
            result.sources,
            result.provider,
            result.model,
            elapsed=time.perf_counter() - t0,
            iterations=result.iterations,
        )
        return result

    # ----- Auditoria ------------------------------------------------------------
    def _append_audit(
        self,
        question: str,
        sources: list[SearchHit],
        provider: str,
        model: str,
        elapsed: float,
        iterations: int = 1,
        cached: bool = False,
    ) -> None:
        path = Path(self._settings.audit_log)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "question": question,
            "provider": provider,
            "model": model,
            "iterations": iterations,
            "elapsed_s": round(elapsed, 3),
            "cache": cached,
            "compression": self._settings.context_compression,
            "sources": [{"doc": s.doc_name, "score": round(s.score, 4)} for s in sources],
        }
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
