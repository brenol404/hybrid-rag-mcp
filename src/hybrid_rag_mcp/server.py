from __future__ import annotations

import argparse
import asyncio
import queue
import threading

from mcp.server.mcpserver import Context, MCPServer

from .config import get_settings
from .rag.engine import RAGEngine

mcp = MCPServer(
    "hybrid-rag-mcp",
    instructions="Busca híbrida + RAG local sobre documentos técnicos com fallback offline.",
)


class _Runtime:
    """Estado compartilhado do servidor, inicializado de forma preguiçosa."""

    engine: RAGEngine | None = None


@mcp.tool()
def ingest(corpus_dir: str = "") -> str:
    """Indexa documentos (md/txt/pdf) de um diretório nos índices vetorial e BM25."""
    engine = _ensure_engine()
    directory = corpus_dir or get_settings().corpus_dir
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
    hits = engine.search(query, top_k=top_k)
    if not hits:
        return "Nenhum resultado encontrado. Execute 'ingest' antes."
    return _format_hits(hits)


@mcp.tool()
async def ask(question: str, context: Context, top_k: int = 5) -> str:
    """Responde com RAG e faz streaming do progresso/tokens via progress notifications."""
    engine = _ensure_engine()
    channel: queue.Queue = queue.Queue()

    def emit(event: str) -> None:
        channel.put(("event", event))

    def on_token(delta: str) -> None:
        channel.put(("token", delta))

    def runner() -> None:
        try:
            result = engine.ask(question, top_k=top_k, on_event=emit, on_tokens=on_token)
            channel.put(("result", result))
        except Exception as exc:  # noqa: BLE001 - resposta de erro volta como texto
            channel.put(("error", str(exc)))

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()

    result = None
    error = None
    token_count = 0
    while True:
        try:
            kind, payload = channel.get(timeout=0.25)
        except queue.Empty:
            if not thread.is_alive() and channel.empty():
                break
            continue
        if kind == "event":
            await context.report_progress(0, None, payload)
        elif kind == "token":
            token_count += 1
            await context.report_progress(token_count, None, payload)
        elif kind == "result":
            result = payload
        elif kind == "error":
            error = payload

    if error is not None:
        return f"Erro: {error}"
    tag = " · cache" if result.cache_hit else ""
    return (
        f"[{result.provider}/{result.model}{tag}]\n{result.answer}\n\n--- Fontes ---\n"
        f"{_format_hits(result.sources)}"
    )


def _ensure_engine() -> RAGEngine:
    if _Runtime.engine is None:
        _Runtime.engine = RAGEngine(get_settings())
    return _Runtime.engine


def _format_hits(hits) -> str:
    lines = []
    for i, h in enumerate(hits, start=1):
        preview = h.content[:160].replace("\n", " ")
        lines.append(f"{i}. [{h.doc_name}] (score={h.score:.3f}) {preview}")
    return "\n".join(lines)


def main() -> None:
    """Ponto de entrada da CLI: serve o MCP via stdio (padrão) ou streamable HTTP."""
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
        asyncio.run(mcp.run_streamable_http_async(host=args.host, port=args.port))
    else:
        asyncio.run(mcp.run_stdio_async())


if __name__ == "__main__":
    main()
