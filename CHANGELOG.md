# Changelog

## [0.2.0] — 2026-10-02

### Interface & Usabilidade

- **Dashboard Visual Web (`hybrid-rag-ui`)**: Nova interface visual interativa (Starlette + Tailwind CSS) para inspeção de busca híbrida em tempo real (RRF vs Vetorial vs Léxico), chat com agente e trilha de passos, e explorador/ingestor de chunks.
- Comando de console `hybrid-rag-ui` registrado em `pyproject.toml`.

### Retrieval & Re-ranking

- **Re-ranking Multi-provedor**: Adicionado suporte neural ao Cohere Rerank API (`rerank-v3.5`, `rerank-multilingual-v3.0`) e endpoints HTTP customizados (Text-Embeddings-Inference / FastAPI), com arquitetura desacoplada `BaseReranker` e degradação graciosa.
- **Fix de Persistência no Windows**: Liberação explícita de descritores de arquivo SQLite/numpy no modo local do Qdrant antes da recriação de coleções por mudança de dimensão.

### Qualidade

- Suíte de testes expandida para 103 testes offline cobrindo endpoints web da UI, novos provedores de re-ranking e isolamento de falhas.
- 100% de conformidade com `ruff check`, `ruff format` e `mypy strict`.

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
