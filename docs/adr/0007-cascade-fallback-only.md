# ADR 0007 — Cascata multi-API: só fallback, sem round-robin/fan-out/GUI

Data: 2026-09

## Contexto

Com `LLM_CHAIN`, era tentador generalizar: round-robin (alternar), fan-out
(perguntar a todas e julgar a melhor) e até interface visual própria.

## Decisão

Só fallback em cascata (ordem = prioridade, Ollama por último). Cortado:

- **round-robin**: existe para distribuir *carga*; nosso volume é baixo e
  esporádico — complexidade (estado compartilhado) sem problema real. Quem
  precisar distribui num gateway (LiteLLM/OpenRouter) na frente do
  `CLOUD_BASE_URL` único.
- **fan-out com juiz**: multiplica custo por N e espera o mais lento, para
  ganho marginal — resposta aqui é *extraída do contexto*, N modelos convergem
  para o mesmo fato. "Melhor de N" pertence ao eval offline (`judge.py`).
- **GUI própria**: escopo de cliente, não de servidor.

## Consequências

- Uma env (`LLM_CHAIN`), um loop, mesma semântica do fallback original.
- Roteamento inteligente continua possível fora do app, via gateway.
