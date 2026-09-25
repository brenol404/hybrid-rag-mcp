# Changelog

## [0.1.0] — 2026-09-25

Primeira release: servidor MCP de RAG híbrido 100% local.

### Retrieval

- Busca híbrida Qdrant (vetorial) + BM25 próprio, fusão RRF com peso léxico 1.5
  (grid-search: recall@1 0.833 → 0.917 em 36 queries).
- Scroll paginado no Qdrant; ingestão incremental idempotente por hash.
- Re-ranking opcional via Ollama com degradação graciosa.

### Geração

- Agente multi-step (`[MORE_CONTEXT]` + memória incremental de fontes).
- Fallback nuvem → Ollama; streaming via progress notifications.
- Cache semântico (append-log + compactação) e compressão estilo-Caveman.

### Qualidade e operação

- 66 testes offline; CI com ruff + mypy strict + cobertura (≥75%) + gate
  `recall@1 >= 0.8`; `pip-audit` e Dependabot.
- llm-as-judge: média 1.92/2 em 12 perguntas (`eval/answers.jsonl`).
- Auth bearer no HTTP; modo Qdrant servidor; rotação do audit log;
  deploy via systemd e Docker Compose.
