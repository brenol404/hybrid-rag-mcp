"""Blocos reutilizáveis do bench de escala (importáveis e testáveis offline).

Ver tools/scale_bench.py para a metodologia completa.
"""

from __future__ import annotations

from .models import DocumentChunk
from .rag.chunker import chunk_text


def build_scale_corpus(
    base_docs: list[tuple[str, str]],
    target_chunks: int,
    chunk_size: int,
    overlap: int,
) -> list[DocumentChunk]:
    """Replica os docs base com salt único até atingir o alvo de chunks.

    Completa sempre a última cópia (overshoot de no máximo 1 cópia): assim
    toda cópia tem o chunk de cauda com salt e nenhum par de docs colide.
    """
    out: list[DocumentChunk] = []
    copy = 0
    while len(out) < target_chunks:
        for doc_name, text in base_docs:
            salted_name = f"scale-{copy:03d}-{doc_name}"
            salted = f"{text}\n\n[escala] cópia {copy} de {doc_name} — token único escala-{copy}.\n"
            out.extend(chunk_text(salted_name, salted, chunk_size=chunk_size, overlap=overlap))
        copy += 1
    return out
