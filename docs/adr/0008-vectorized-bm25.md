# ADR 0008 — BM25 vetorizado: aceitar fit mais lento por busca mais rápida

Data: 2026-09

## Contexto

O bench apontou o BM25 em Python puro como suspeito de gargalo. Medido: eram
~17ms de ~400ms por busca (~4%) — o palpite estava errado, o grosso é
embedding no Ollama + HNSW local.

## Decisão

Vetorizar mesmo assim (postings + numpy, paridade < 1e-9 com força bruta):
`score_all` 17,5→1,7ms (~10x). Trade registrado: `fit` ~3x mais lento
(construção dos postings), pago 1x por ingest contra N buscas.

## Consequências

- O ganho compõe com escala (a 10x de corpus, o loop puro dominaria).
- numpy vira dependência direta; `_tfs`/`_dl` removidos (sem dupla
  representação em memória).
- README registra a refutação, não só o speedup.
