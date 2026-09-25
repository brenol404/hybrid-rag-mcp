from __future__ import annotations

import argparse
import asyncio
import logging
import queue
import secrets
import sys
import threading
from collections.abc import Awaitable, Callable
from pathlib import Path

from mcp.server.mcpserver import Context, MCPServer
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Route
from starlette.types import ASGIApp

from .config import Settings, get_settings
from .models import AskResult, SearchHit
from .rag.engine import RAGEngine

mcp = MCPServer(
    "hybrid-rag-mcp",
    instructions="Busca híbrida + RAG local sobre documentos técnicos com fallback offline.",
)

_MAX_TOP_K = 50
_DEFAULT_TOP_K = 5


def _clamp_top_k(top_k: int | None, default: int = _DEFAULT_TOP_K) -> int:
    """Limita `top_k` ao intervalo [1, _MAX_TOP_K].

    Valores absurdos vindos da rede não devem chegar ao Qdrant (negativo
    quebra a query; gigante pesa/estoura a busca).
    """
    if top_k is None:
        return default
    return max(1, min(int(top_k), _MAX_TOP_K))


def _resolve_corpus_dir(corpus_dir: str, default_dir: str) -> str:
    """Valida o diretório apontado por `ingest` antes de tocar no engine."""
    directory = corpus_dir or default_dir
    if not Path(directory).is_dir():
        raise ValueError(f"Diretório de corpus não encontrado ou não é um diretório: {directory}")
    return directory


class _Runtime:
    """Estado compartilhado do servidor, inicializado de forma preguiçosa."""

    engine: RAGEngine | None = None


@mcp.tool()
def ingest(corpus_dir: str = "") -> str:
    """Indexa documentos (md/txt/pdf) de um diretório nos índices vetorial e BM25."""
    engine = _ensure_engine()
    directory = _resolve_corpus_dir(corpus_dir, get_settings().corpus_dir)
    stats = engine.ingest(directory)
    return (
        f"Indexado: {stats['documents']} documentos, {stats['chunks']} chunks "
        f"(+{stats['added']} novos, {stats['unchanged']} inalterados, "
        f"-{stats['deleted']} órfãos; {stats['indexed']} no índice)."
    )


@mcp.tool()
def search(query: str, top_k: int = 5) -> str:
    """Busca híbrida (vetorial + BM25 via RRF) no corpus indexado. Retorna trechos e fontes."""
    engine = _ensure_engine()
    hits = engine.search(query, top_k=_clamp_top_k(top_k))
    if not hits:
        return "Nenhum resultado encontrado. Execute 'ingest' antes."
    return _format_hits(hits)


@mcp.tool()
def metrics() -> str:
    """Resumo das métricas do servidor (contadores + latências p50/p95 desde o boot)."""
    return _ensure_engine().metrics.render_text()


@mcp.tool()
async def ask(question: str, context: Context, top_k: int = 5) -> str:
    """Responde com RAG e faz streaming do progresso/tokens via progress notifications."""
    engine = _ensure_engine()
    channel: queue.Queue[tuple[str, str | AskResult]] = queue.Queue()

    def emit(event: str) -> None:
        channel.put(("event", event))

    def on_token(delta: str) -> None:
        channel.put(("token", delta))

    def runner() -> None:
        try:
            result = engine.ask(
                question, top_k=_clamp_top_k(top_k), on_event=emit, on_tokens=on_token
            )
            channel.put(("result", result))
        except Exception as exc:  # noqa: BLE001 - resposta de erro volta como texto
            channel.put(("error", str(exc)))

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()

    result: AskResult | None = None
    error: str | None = None
    token_count = 0
    while True:
        try:
            kind, payload = channel.get(timeout=0.25)
        except queue.Empty:
            if not thread.is_alive() and channel.empty():
                break
            continue
        if kind == "event":
            assert isinstance(payload, str)
            await context.report_progress(0, None, payload)
        elif kind == "token":
            assert isinstance(payload, str)
            token_count += 1
            await context.report_progress(token_count, None, payload)
        elif kind == "result":
            assert isinstance(payload, AskResult)
            result = payload
        elif kind == "error":
            assert isinstance(payload, str)
            error = payload

    if error is not None:
        return f"Erro: {error}"
    assert result is not None  # runner sempre entrega result ou error
    tag = " · cache" if result.cache_hit else ""
    return (
        f"[{result.provider}/{result.model}{tag}]\n{result.answer}\n\n--- Fontes ---\n"
        f"{_format_hits(result.sources)}"
    )


def _ensure_engine() -> RAGEngine:
    if _Runtime.engine is None:
        _Runtime.engine = RAGEngine(get_settings())
    return _Runtime.engine


def _format_hits(hits: list[SearchHit]) -> str:
    lines = []
    for i, h in enumerate(hits, start=1):
        preview = h.content[:160].replace("\n", " ")
        lines.append(f"{i}. [{h.doc_name}] (score={h.score:.3f}) {preview}")
    return "\n".join(lines)


class _BearerAuthMiddleware(BaseHTTPMiddleware):
    """Exige `Authorization: Bearer <token>` em todo o HTTP quando configurado.

    Sem token configurado o middleware nem entra na pilha (comportamento
    idêntico ao `run_streamable_http_async` do SDK). Comparação em tempo
    constante contra timing attack.
    """

    def __init__(self, app: ASGIApp, token: str) -> None:
        super().__init__(app)
        self._token = token

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        auth = request.headers.get("authorization", "")
        if not secrets.compare_digest(auth, f"Bearer {self._token}"):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)


def _build_http_app(settings: Settings) -> Starlette:
    """Monta o app streamable HTTP do SDK, com auth opcional por bearer token."""
    app = mcp.streamable_http_app()
    # O SDK não é tipado (retorno Any): checagem em runtime, não cast cego.
    assert isinstance(app, Starlette)
    if settings.mcp_auth_token:
        app.add_middleware(_BearerAuthMiddleware, token=settings.mcp_auth_token)
    app.router.routes.append(Route("/metrics", _metrics_endpoint))
    return app


async def _metrics_endpoint(request: Request) -> PlainTextResponse:
    """Métricas em formato Prometheus (herda o bearer auth do app)."""
    return PlainTextResponse(_ensure_engine().metrics.render_prometheus())


async def _run_http(host: str, port: int) -> None:
    import uvicorn

    app = _build_http_app(get_settings())
    config = uvicorn.Config(app, host=host, port=port, log_level=mcp.settings.log_level.lower())
    await uvicorn.Server(config).serve()


def _configure_logging() -> None:
    """Logs JSON do engine no stderr (o stdout pertence ao protocolo MCP no stdio).

    Configura o logger do pacote de forma explícita em vez de `basicConfig`:
    a raiz já costuma ter handlers do SDK/uvicorn, e `basicConfig` vira no-op
    nesse caso (nível INFO nunca aplicado).
    """
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger("hybrid_rag_mcp")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # uma cópia só, sem duplicar nos handlers da raiz


def main() -> None:
    """Ponto de entrada da CLI: serve o MCP via stdio (padrão) ou streamable HTTP."""
    _configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="stdio: RPC local; http: endpoint MCP streamable (http://host:port/mcp)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.transport == "http":
        asyncio.run(_run_http(args.host, args.port))
    else:
        asyncio.run(mcp.run_stdio_async())


if __name__ == "__main__":
    main()
