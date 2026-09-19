from __future__ import annotations

from pathlib import Path

from ..config import Settings
from ..models import DocumentChunk
from ..providers import EmbeddingProvider
from ..stores import LexicalStore, VectorStore
from .chunker import chunk_text

SUPPORTED_EXTS = {".md", ".txt", ".pdf"}


def read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="replace")


def ingest_directory(
    corpus_dir: str,
    settings: Settings,
    vector_store: VectorStore,
    lexical_store: LexicalStore,
    embedder: EmbeddingProvider,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> dict[str, int]:
    """Indexa todos os documentos suportados do diretório nos dois índices."""
    root = Path(corpus_dir)
    if not root.exists():
        raise FileNotFoundError(f"Diretório de corpus não encontrado: {corpus_dir}")

    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap
    stats: dict[str, int] = {}
    total_chunks: list[DocumentChunk] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTS:
            continue
        text = read_document(path)
        chunks = chunk_text(path.name, text, chunk_size=chunk_size, overlap=overlap)
        total_chunks.extend(chunks)
        stats[path.name] = len(chunks)

    # Sincronização incremental: embed somente chunks novos, poda órfãos.
    sync = vector_store.sync_chunks(total_chunks)
    stored = vector_store.all_chunks()
    lexical_store.rebuild(stored)

    return {
        "documents": len(stats),
        "chunks": len(total_chunks),
        "added": sync["added"],
        "deleted": sync["deleted"],
        "unchanged": sync["unchanged"],
        "indexed": len(stored),
        "per_doc": stats,
    }
