"""Cliente MCP de exemplo: fala com o servidor hybrid-rag-mcp pela stdio.

Uso:
    python examples/client.py  (responde uma pergunta fixa)
    python examples/client.py "Qual a porta padrão do servidor?"
"""

from __future__ import annotations

import asyncio
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def run(question: str) -> None:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "hybrid_rag_mcp"],
        env={**os.environ, "PYTHONPATH": "src"},
    )

    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = await session.list_tools()
        print(f"Ferramentas disponíveis: {', '.join(t.name for t in tools.tools)}\n")

        await session.call_tool("ingest", {})
        result = await session.call_tool("ask", {"question": question})
        text = result.content[0].text
        print(text)


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "Qual a porta padrão do servidor?"
    asyncio.run(run(q))
