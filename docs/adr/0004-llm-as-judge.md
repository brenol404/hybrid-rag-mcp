# ADR 0004 — Avaliação de respostas via llm-as-judge (NOTA 0/1/2)

Data: 2026-09 (commit `d8147a0`)

## Contexto

O gate de retrieval (`recall@1 >= 0.8`) prova que o trecho certo sobe, mas não
que a resposta final está correta — um `ask` pode alucinar com contexto certo.

## Decisão

`eval/answers.jsonl` (pergunta + resposta esperada aterrada no corpus) +
`src/hybrid_rag_mcp/judge.py`: o próprio Ollama dá NOTA 0/1/2 com
justificativa. Regras anti-autoengano:

- sem `NOTA:` parseável → "sem veredito", **nunca** acerto silencioso;
- erro de geração → item sem veredito, não zero;
- média calculada só sobre itens avaliados.

## Consequências

- Primeira medida real: média **1.92/2** (11×2, 1×1) — publicada no README.
- A nota depende do modelo-juiz: não é gate do CI (não-reprodutível entre
  versões de modelo), é instrumento de regressão manual.
