# ADR 0005 — Inicialização preguiçosa do engine + `warmup()` explícito

Data: 2026-09 (commits `ac122f5`, `07be790`)

## Contexto

Construir o `RAGEngine` consultava o Ollama (dim do embedding) e restaurava o
BM25 — qualquer falha de startup derrubava o primeiro uso, e o servidor MCP
pagava esse custo por/tool call fria.

## Decisão

Construção barata (zero I/O): dim resolvida sob demanda e cacheada, coleção
criada no 1º uso, BM25 restaurado no 1º `search`/`ask` (cache-hit nem restaura).
Quem usa os stores internos direto (`tools/grid_search.py`) chama `warmup()`.

## Consequências

- A preguiça quase causou regressão silenciosa: o grid lia o léxico vazio
  (0 chunks) e rodaria o tuning só no vetorial. Foi pega por teste de
  regressão (`test_engine_warmup_restaura_lexico_para_uso_direto`).
- Contrato explícito > convenção implícita: `warmup()` documenta quem precisa
  dele em vez de confiar na ordem de chamadas.
