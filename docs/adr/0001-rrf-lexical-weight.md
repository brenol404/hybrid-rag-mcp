# ADR 0001 — Peso léxico 1.5 na fusão RRF

Data: 2026-09 (commit `2596f33`)

## Contexto

A fusão RRF combina busca vetorial (Qdrant) e léxica (BM25 próprio) com pesos
`(w_vec, w_lex)`. O default "puro" seria (1.0, 1.0).

## Decisão

Fixar `(1.0, 1.5)` como default (`RRF_W_VECTOR`, `RRF_W_LEXICAL`).

## Evidência

`tools/grid_search.py` varreu rrf_k × bm25_top_k × pesos sobre as 36 consultas
do eval: recall@1 saiu de **0.833 → 0.917** com peso léxico 1.5. Histórico em
`eval/grid_results.json`.

## Consequências

- Corpus técnico com termos exatos (portas, nomes de serviço) favorece o lado
  léxico; o peso compensa o viés do embedding para paráfrases.
- Se o corpus mudar de perfil, rodar o grid de novo em vez de chutar.
