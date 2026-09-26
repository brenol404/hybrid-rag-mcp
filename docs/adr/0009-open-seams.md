# ADR 0009 — Abertura sem framework: costuras + docs, nada obrigatório

Data: 2026-09

## Contexto

Filosofia do projeto: usuário tem livre-arbítrio (trocar compressor,
embeddings, reranker) sem forkar. Tentação oposta: framework de plugins
com registry e entry points.

## Decisão

Costuras mínimas + documentação, sem framework:

- compressor: qualquer callable via `CONTEXT_COMPRESSOR=pkg:func` (vence níveis);
- embeddings/LLM: ABCs existentes + `CLOUD_*` (nuvem opt-in, Ollama default);
- reranker: duck-type `score(query, texts)`;
- `llms.txt` + `SKILL.md` para consumo por agentes.

Defaults intocados em tudo — abrir não pode impor nada a quem já usa.

## Consequências

- Framework seria abstração prematura; revisor chamaria de over-engineering.
- `CloudEmbeddings` prova a costura; `QDRANT_RECREATE_ON_DIM_CHANGE` existe
  porque trocar embedding invalida o índice (decisão explícita, nunca wipe
  silencioso).
