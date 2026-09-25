# ADR 0002 — Qdrant embarcado como default, servidor como opt-in

Data: 2026-09 (commits `1f6ee28`, `793d9d3`)

## Contexto

Precisamos de busca vetorial persistente. Opções: (a) Qdrant embarcado (modo
local, sem Docker); (b) exigir servidor Qdrant via Docker desde o dia 1.

## Decisão

Embarcado como default (`QDRANT_PATH=data/qdrant`); servidor como opt-in
(`QDRANT_URL=http://localhost:6333` + `docker-compose.yml`).

## Consequências

- `pip install` basta para começar; zero infra para contribuir.
- Embarcado = 1 processo por vez (lock de arquivo). Sessões simultâneas
  exigem o modo servidor — documentado no README (Deploy).
- Todo o acesso passa por `VectorStore`, então os dois modos compartilham
  sync, scroll paginado e persistência do BM25.
