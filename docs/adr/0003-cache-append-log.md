# ADR 0003 — Cache semântico em append-log com compactação periódica

Data: 2026-09 (commits `15ad979`, `ac122f5`)

## Contexto

O cache semântico (`data/cache.jsonl`) reescrevia o arquivo inteiro a cada
`store` — O(n) de I/O por `ask`, crescendo com `CACHE_MAX_ENTRIES`.

## Decisão

Escrita em append-log (uma linha por gravação, O(1)) + compactação completa a
cada 64 gravações + teto de `max_entries` aplicado também na carga (o arquivo
pode ficar temporariamente maior entre compactações, a memória não).

## Consequências

- `store` não degrada com o teto; reloads deduplicam por pergunta (vale a mais
  recente) e respeitam o mesmo teto da memória — disco e memória consistentes.
- Trade-off aceito: linhas obsoletas entre compactações (limitado a 64).
