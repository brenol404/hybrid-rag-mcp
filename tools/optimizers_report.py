"""Relatório e hub de otimizadores de tokens do projeto.

Uso:
    python tools/optimizers_report.py [--json]

Lista o que está ativo no runtime (cache semântico, compressão de contexto),
estima a economia atual e documenta as integrações externas avaliadas
(RTK/Headroom/Caveman/Ponytail) com o comando de instalação futuro de cada uma.
Nada é instalado ou modificado — é apenas um painel informativo.

Para ativar nas próximas execuções:
  - cache+compressão já embutidos: configure CONTEXT_COMPRESSION no .env
  - Headroom (compressão reversível / MCP sidecar) e demais: ops de integração
    listadas no JSON abaixo; nenhuma dependência extra entra no core.
"""

from __future__ import annotations

import json
import sys

from hybrid_rag_mcp.config import Settings
from hybrid_rag_mcp.optimize import make_compressor, token_savings


def _report() -> dict:
    s = Settings()
    compressor = make_compressor(s.context_compression)
    level_names = {0: "off", 1: "leve", 2: "agressivo"}
    sample = (
        "O servidor usa Redis e Qdrant, porque ambos são locais e gratuitos. "
        "A janela de manutenção é das 02h às 04h."
    )
    compressed = compressor(sample) if compressor else sample
    savings = token_savings(sample, compressed)

    return {
        "runtime": {
            "semantic_cache": {
                "enabled": s.cache_enabled,
                "path": s.cache_path,
                "sim_threshold": s.cache_sim_threshold,
                "ttl_sec": s.cache_ttl_sec,
                "max_entries": s.cache_max_entries,
            },
            "context_compression": {
                "level": s.context_compression,
                "level_name": level_names.get(s.context_compression, "?"),
                "sample_savings": savings,
                "sample_compressed": compressed,
            },
        },
        "integrations_avaliadas": [
            {
                "nome": "Headroom (headroomlabs-ai/headroom)",
                "onde": "compressão reversível de contexto, tool outputs e RAG chunks; lib Python e MCP server próprios",
                "status": "futuro — carga de ONNX/HF; roda como sidecar MCP",
                "instalar": "pip install headroom  # só quando adotarmos compressão externa",
            },
            {
                "nome": "RTK (rtk-ai/rtk)",
                "onde": "compressão da saída de shell para agentes de coding (fora do nosso servidor RAG)",
                "status": "fora do escopo do runtime — recomendado no ambiente de dev de agentes",
                "instalar": "curl -sSL https://rtk.ai/install | bash  # fora do projeto",
            },
            {
                "nome": "Caveman (wilpel/caveman-compression)",
                "onde": "princípio (tirar gramática previsível) já embutido em CONTEXT_COMPRESSION",
                "status": "implementado como compressor determinístico próprio (PT/EN, zero deps)",
                "instalar": "n/a — já embutido",
            },
            {
                "nome": "Ponytail (DietrichGebert/ponytail)",
                "onde": "reduz o volume de código que agentes escrevem (não diminui contexto de RAG)",
                "status": "fora do escopo — filosofia de agente de coding",
                "instalar": "n/a — não se aplica a este servidor",
            },
        ],
    }


def main() -> int:
    rep = _report()
    if "--json" in sys.argv:
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        return 0

    rt = rep["runtime"]
    print("== Otimizadores de tokens — runtime ==")
    sc = rt["semantic_cache"]
    print(f"- Cache semântico: {sc['enabled']}  ({sc['path']}, sim>={sc['sim_threshold']}, ttl={sc['ttl_sec']}s)")
    c = rt["context_compression"]
    print(f"- Compressão de contexto: {c['level_name']} (nível {c['level']})  "
          f"— amostra economiza {c['sample_savings'] * 100:.0f}% dos tokens")
    if c["level"]:
        print(f"    ex.: {c['sample_compressed']}")
    print("\n== Integrações externas avaliadas (futuro) ==")
    for item in rep["integrations_avaliadas"]:
        print(f"- [{item['nome']}] {item['onde']}\n    status: {item['status']}\n    instalar: {item['instalar']}")

    print("\nPara ligar a compressão agora, configure CONTEXT_COMPRESSION=1|2 no .env.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())