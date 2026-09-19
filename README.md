# hybrid-rag-mcp

[![CI](https://github.com/brenol404/hybrid-rag-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/brenol404/hybrid-rag-mcp/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Serve MCP com **RAG híbrido** (Qdrant vetorial + BM25 léxico via RRF), **agente multi-step** com fallback offline via **Ollama** e transporte **stdio** ou **streamable HTTP**.

Foco: indexar documentos técnicos (Markdown, TXT, PDF) e responder perguntas com **fontes citadas**, de forma **100% local** — o padrão que conecta LLMs a bases locais/corporativas em 2026/2027.

## Destaques

- **Busca híbrida**: embedding (Qdrant local, sem Docker) + BM25 próprio (idf suavizado), fundidos por **RRF**.
- **Agente multi-step**: se o contexto da 1ª busca for insuficiente, o modelo sinaliza `[MORE_CONTEXT]`, o agente gera uma busca de follow-up e repete com **memória incremental de fontes**.
- **Re-ranking opcional**: cross-encoder (Ollama `/api/rerank`, ex. `bge-reranker-v2-m3`) com *degradação graciosa*.
- **Fallback resiliente**: provedor de nuvem (OpenAI-compatible) na frente, **Ollama local como reserva** quando a API cai.
- **Persistência**: os chunks ficam no Qdrant; o índice BM25 é **restaurado no startup** sem re-ingestão.
- **Ingestão incremental**: re-rodar `ingest` só embeda o que mudou (idempotente por hash de conteúdo) e poda órfãos — barato em CI e em re-deploys.
- **Rastreabilidade**: `trace` por passo do agente + `audit.jsonl` (pergunta, provedor, iterações, latência, fontes).
- **Mensurável**: pipeline de avaliação `recall@k` / `nDCG@k` com **gate de qualidade no CI**.
- **Dois transportes**: stdio (RPC local) e **streamable HTTP** (`http://host:port/mcp`).

## Arquitetura

```mermaid
flowchart LR
    C[Cliente MCP<br/>stdio ou HTTP] -->|tools: ingest / search / ask| M[MCP Server<br/>hybrid-rag-mcp]
    M --> AGE[Agente multi-step<br/>loop com [MORE_CONTEXT]]
    M --> I[ingest]
    I --> C1[Chunker<br/>seções + sentenças]
    C1 --> E[Embeddings<br/>Ollama bge-m3]
    E --> Q1[(Qdrant local<br/>busca vetorial)]
    C1 --> K[BM25 próprio<br/>busca léxica]
    AGE --> RET[Busca híbrida]
    RET --> Q1 & K
    Q1 & K --> RRF[RRF fusion]
    RRF --> RR[Reranker opcional<br/>Ollama /api/rerank]
    RR --> LLM[FallbackLLM<br/>nuvem -> Ollama]
    LLM --> AUD[audit.jsonl<br/>trace + iterações + fontes]
```

## Métricas (gate de qualidade no CI)

Pipeline de avaliação sobre **36 queries** — 12 curadas manualmente em `eval/dataset.jsonl` + 24 geradas automaticamente de documentos reais (`eval/dataset.real.jsonl`, veja [Corpus](#corpus)).
Gatilho do CI: **falha se `recall@1 < 0.8`**.

| k  | recall@k | nDCG@k |
|----|----------|--------|
| 1  | **0.833** | 0.833 |
| 3  | **1.000** | 0.836 |
| 5  | **1.000** | 0.907 |

Números honestos sobre texto real: `recall@1 = 0.833`, mas a fonte certa está sempre no top-3. Rode localmente com `python -m hybrid_rag_mcp.eval`.

## Corpus

- `examples/corpus/` — **11 documentos**: 5 fictícios (operations/security/database/infra/events) + 6 livros reais de domínio público (Project Gutenberg): Chekhov, Machado de Assis, Aluísio Azevedo, Eça de Queirós, Jane Austen e Conan Doyle, misturando PT e EN.
- `tools/fetch_corpus.py` — baixa o catálogo do Gutenberg e **gera o dataset de avaliação automaticamente**: cada consulta é um trecho real do documento e o documento esperado é a fonte exata desse trecho (ground-truth auto-supervisionado, sem curadoria manual).

## Como rodar

Pré-requisitos: **Python 3.11+**, **Ollama** de pé (`ollama serve`).

```bash
# 1. Modelos locais (uma vez)
ollama pull bge-m3        # embeddings
ollama pull qwen3:8b      # geração (ou outro)
ollama pull bge-reranker-v2-m3   # opcional: somente Ollama >= 0.36

# 2. Instalar
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 3. Indexar + responder (uso direto da engine)
python -c "
from hybrid_rag_mcp.rag.engine import RAGEngine
rag = RAGEngine()
print(rag.ingest())                                    # indexa examples/corpus
print(rag.search('Qual a porta padrão do servidor?'))  # busca híbrida
print(rag.ask('De quantas em quantas horas são os backups?'))  # agente com fontes
"
```

### Como cliente MCP

**stdio:** cliente de exemplo — `python examples/client.py "Qual a porta padrão?"`

**HTTP:**

```bash
# terminal 1
python -m hybrid_rag_mcp --transport http --host 127.0.0.1 --port 8000

# terminal 2
python examples/client_http.py "Qual a porta padrão?"
```

Registre em qualquer cliente MCP (Claude Desktop, editores, agentes):

```json
{
  "mcpServers": {
    "hybrid-rag": {
      "command": ".venv/bin/python",
      "args": ["-m", "hybrid_rag_mcp"],
      "env": { "PYTHONPATH": "src" }
    }
  }
}
```

### Fallback para nuvem (opcional)

Copie `.env.example` para `.env` e preencha `CLOUD_BASE_URL` + `CLOUD_API_KEY` + `CLOUD_MODEL`
(qualquer endpoint OpenAI-compatível). A nuvem assume prioridade; o Ollama responde automaticamente
se a API falhar ou ficar offline.

## Ferramentas MCP

| Tool    | Descrição |
|---------|-----------|
| `ingest` | Indexa `md`/`txt`/`pdf` de um diretório nos dois índices (Qdrant + BM25). Incremental: só re-embeda chunks alterados. |
| `search` | Busca híbrida (RRF, com re-ranking opcional) retornando trechos + fontes. |
| `ask`    | Agente multi-step: recupera, gera, detecta contexto insuficiente, refaz a busca e responde citando fontes (com audit log). |

## Estrutura

```
src/hybrid_rag_mcp/
├── server.py          # Servidor MCP (stdio + streamable HTTP)
├── config.py          # Configuração via .env (pydantic-settings)
├── eval.py            # Avaliação recall@k / nDCG@k
├── rag/
│   ├── agent.py       # Loop multi-step (memória de fontes, [MORE_CONTEXT])
│   ├── chunker.py     # Chunking por seções markdown + sentenças
│   ├── engine.py      # Orquestração: ingest → search → ask + audit
│   ├── hybrid.py      # Fusão RRF + re-ranking
│   └── ingestion.py   # Leitura de md/txt/pdf
├── providers/
│   ├── embed.py       # Embeddings via Ollama
│   ├── llm.py         # FallbackLLM (nuvem → Ollama)
│   └── rerank.py      # Cross-encoder opcional (degradação graciosa)
└── stores/
    ├── vector.py      # Qdrant embarcado (sem Docker, persistente)
    └── lexic.py       # BM25 com idf suavizado
```

## Qualidade

- **21 testes unitários** (`pytest`) sem rede/Ollama — chunking, RRF, BM25, persistência, métricas de eval e loop do agente.
- CI em 2 jobs: `test` (ruff + pytest + smoke stdio/HTTP) e `eval` (Ollama real + gate `recall@1 >= 0.8`).
- `docker-compose.yml` intencionalmente ausente: roda só com `pip install` (Qdrant embarcado).