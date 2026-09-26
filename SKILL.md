---
name: hybrid-rag-mcp
description: Local hybrid-RAG MCP server (Qdrant + BM25). Index docs and answer questions with cited sources, fully offline.
---

# hybrid-rag-mcp skill

Use these tools when the user asks about indexed documents or wants RAG answers
with sources. Everything runs locally via Ollama (no account needed).

## Start (one time per session need)

- stdio is automatic for local clients; for HTTP: `python -m hybrid_rag_mcp --transport http --port 8000`
- Models needed in Ollama: `bge-m3` (embeddings), `qwen3:8b` (generation)

## Tools

- `ingest` (no args = default corpus): index/re-index documents. Incremental — only changed chunks re-embed.
- `search(query, top_k)`: hybrid retrieval. Returns snippets with `[doc]` sources and scores.
- `ask(question, top_k)`: full RAG answer with inline citations like `[manual.pdf]`. Checks the semantic cache first; multi-step on thin context.
- `metrics()`: counters + p50/p95 latencies since boot. Use to check health/load.

## Rules

- Cite sources exactly as the tool returns them (`[doc-name]`).
- If `search` returns nothing useful, call `ingest` first — the index may be empty or stale.
- `top_k` is clamped to [1, 50] server-side.
- With `MCP_AUTH_TOKEN` set, send `Authorization: Bearer <token>`.
