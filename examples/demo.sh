#!/usr/bin/env bash
# Demo de ~1 min do hybrid-rag-mcp: ingest + search + ask sobre o corpus de exemplo.
# Requer: venv instalada (pip install -e ".[dev]") e Ollama com bge-m3 + qwen3:8b.
# Uso: bash examples/demo.sh
set -euo pipefail
cd "$(dirname "$0")/.."

python - <<'EOF'
from hybrid_rag_mcp.config import get_settings
from hybrid_rag_mcp.rag.engine import RAGEngine

rag = RAGEngine(get_settings())
stats = rag.ingest()
print(f"[ingest] {stats['documents']} docs, {stats['chunks']} chunks "
      f"(+{stats['added']} novos, {stats['indexed']} no índice)")

q = "Qual a porta padrão do servidor?"
hits = rag.search(q, top_k=3)
print(f"[search] '{q}'")
for h in hits:
    print(f"  - [{h.doc_name}] score={h.score:.3f}: {h.content[:90]}...")

a = "De quantas em quantas horas são os backups?"
res = rag.ask(a)
print(f"[ask] '{a}'\n  [{res.provider}/{res.model}] {res.answer[:220]}")
print(f"  fontes: {', '.join(sorted({s.doc_name for s in res.sources}))}")
rag.close()
EOF
