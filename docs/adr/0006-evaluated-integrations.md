# ADR 0006 — Integrações externas de otimização avaliadas (nada entra no core)

Data: 2026-09

## Contexto

Avaliamos 4 ferramentas externas de otimização de tokens/contexto para decidir
o que adotar, adaptar ou descartar — sem adicionar dependências ao runtime.

## Avaliação

| Ferramenta | Onde atuaria | Veredito |
|---|---|---|
| Headroom (headroomlabs-ai/headroom) | compressão reversível de contexto/tool outputs/RAG chunks; lib Python + MCP server próprios | adotável futuramente como sidecar MCP; carga ONNX/HF (`pip install headroom`) |
| RTK (rtk-ai/rtk) | compressão de saída de shell para agentes de coding | fora do runtime — recomendado no ambiente de dev |
| Caveman (wilpel/caveman-compression) | princípio "tirar gramática, manter fatos" (PT incluso) | **adotado como princípio**: reimplementado em `CONTEXT_COMPRESSION`, zero deps |
| Ponytail (DietrichGebert/ponytail) | cortar volume de código gerado por agentes | não se aplica a um servidor RAG |

## Decisão

Nada entra no core como dependência. O único aproveitamento é o princípio do
Caveman, reimplementado de forma determinística e multilíngue (PT/EN).

## Consequências

- Core segue `pip install` puro, sem ONNX/HF/modelos externos.
- Se compressão reversível fizer falta um dia, Headroom é o candidato
  (sidecar MCP, fora do processo principal).
