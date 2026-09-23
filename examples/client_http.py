"""Cliente MCP via streamable HTTP.

1) Suba o servidor:
    python -m hybrid_rag_mcp --transport http --host 127.0.0.1 --port 8000

2) Noutro terminal:
    python examples/client_http.py "Qual a porta padrão do servidor?"

URL padrão: http://127.0.0.1:8000/mcp

Com auth ligada no servidor (MCP_AUTH_TOKEN), exporte o mesmo token:
    MCP_AUTH_TOKEN=... python examples/client_http.py "pergunta"
"""

from __future__ import annotations

import asyncio
import os
import sys

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


async def run(question: str, url: str) -> None:
    headers = {}
    if os.environ.get("MCP_AUTH_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['MCP_AUTH_TOKEN']}"
    client = httpx.AsyncClient(headers=headers, timeout=60)
    async with (
        streamable_http_client(url, http_client=client) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        tools = await session.list_tools()
        print(f"Ferramentas: {', '.join(t.name for t in tools.tools)}\n")

        await session.call_tool("ingest", {})
        result = await session.call_tool("ask", {"question": question})
        print(result.content[0].text)


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "Qual a porta padrão do servidor?"
    url = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8000/mcp"
    asyncio.run(run(q, url))
