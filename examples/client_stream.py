"""Smoke do streaming de progresso do MCP: sobe o servidor via stdio e chama `ask`
com progress_callback, registrando as notificações recebidas em tempo real."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]


async def main() -> None:
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "hybrid_rag_mcp"],
        env={**os.environ, "PYTHONPATH": "src"},
        cwd=str(ROOT),
    )
    events: list[dict] = []

    def progress_cb(progress: float, total: float | None, request_id=None):
        events.append({"progress": progress, "total": total})

    async with stdio_client(server) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        try:
            result = await session.call_tool(
                "ask",
                {"question": "Qual a política de manutenção do banco PostgreSQL?"},
                progress_callback=progress_cb,
            )
        except (RuntimeError, ConnectionError) as exc:
            print("ERRO:", exc)
            return

    print("progress notifications recebidas:", len(events))
    print("com total informado:", sum(1 for e in events if e["total"] is not None))
    for t in result.content:
        if hasattr(t, "text"):
            print("RESPOSTA (primeiras 200 chars):", t.text[:200].replace("\n", " "))

    print("FIM", "ok" if events else "sem progresso (cliente sem token?)")


if __name__ == "__main__":
    asyncio.run(main())
