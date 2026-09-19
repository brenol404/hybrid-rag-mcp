from __future__ import annotations

import re

from ..models import DocumentChunk

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_HEADING = re.compile(r"^(#{1,6}\s|[-=*]{3,}\s*$|^[A-Z][A-Za-zÀ-ÿ ]{3,}:\s*\n)")


def chunk_text(
    doc_name: str, text: str, chunk_size: int = 512, overlap: int = 64
) -> list[DocumentChunk]:
    """Divide o documento em seções (cabeçalhos markdown) e depois em sentenças."""
    if not text.strip():
        return []

    chunks: list[DocumentChunk] = []
    index = 0

    for section in _split_sections(text):
        buffer = ""
        for sent in _sentences(section):
            if buffer and len(buffer) + len(sent) + 1 > chunk_size:
                chunks.append(_make_chunk(doc_name, index, buffer))
                index += 1
                buffer = _tail(buffer, overlap)
            buffer += sent + " "
        if buffer.strip():
            chunks.append(_make_chunk(doc_name, index, buffer))
            index += 1

    return dedupe(chunks)


def _split_sections(text: str) -> list[str]:
    """Quebra o texto em seções por linhas de cabeçalho (#, ~~~, 'Termo:' etc.)."""
    lines = text.splitlines()
    sections: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            sections.append("\n".join(current))
            current.clear()

    for line in lines:
        if re.match(r"^#{1,6}\s", line) or re.match(r"^={3,}|^-{3,}$", line.strip()):
            flush()
        current.append(line)
        if re.match(r"^={3,}|^-{3,}$", line.strip()):
            flush()
    flush()
    return [s for s in sections if s.strip()]


def _sentences(section: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(section) if s.strip()]


def _tail(text: str, overlap: int) -> str:
    """Recupera as últimas ~overlap chars cortando em fronteira de palavra."""
    if len(text) <= overlap:
        return text
    tail = text[-overlap:]
    first_word = tail.split(" ", 1)[0]
    return tail[len(first_word) :].lstrip() if " " in tail else tail


def dedupe(chunks: list[DocumentChunk]) -> list[DocumentChunk]:
    seen: set[str] = set()
    out: list[DocumentChunk] = []
    for c in chunks:
        if c.content not in seen:
            seen.add(c.content)
            out.append(c)
    return out


def _make_chunk(doc_name: str, index: int, content: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"{doc_name}::chunk::{index}",
        doc_name=doc_name,
        content=content.strip(),
        index=index,
    )
