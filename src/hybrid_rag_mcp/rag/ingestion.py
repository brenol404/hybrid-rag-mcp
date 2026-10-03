from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from ..config import Settings
from ..models import DocumentChunk
from ..providers import EmbeddingProvider
from ..stores import LexicalStore, VectorStore
from .chunker import chunk_text

SUPPORTED_EXTS = {
    ".md",
    ".txt",
    ".pdf",
    ".docx",
    ".csv",
    ".tsv",
    ".html",
    ".htm",
    ".json",
    ".jsonl",
}


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _read_docx(path: Path) -> str:
    """Extrai texto e tabelas de arquivos .docx via zipfile/xml nativo sem dependências externas."""
    paragraphs: list[str] = []
    try:
        with zipfile.ZipFile(path) as z:
            if "word/document.xml" not in z.namelist():
                return ""
            xml_content = z.read("word/document.xml")
            tree = ET.fromstring(xml_content)
            ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
            for p in tree.iter(f"{{{ns['w']}}}p"):
                texts = [node.text for node in p.iter(f"{{{ns['w']}}}t") if node.text]
                if texts:
                    paragraphs.append("".join(texts))
    except Exception:  # noqa: BLE001
        return ""
    return "\n\n".join(paragraphs)


def _read_csv_tsv(path: Path) -> str:
    """Formata linhas tabulares (CSV/TSV) preservando o cabeçalho em formato chave-valor."""
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    content = path.read_text(encoding="utf-8", errors="replace")
    lines = content.splitlines()
    if not lines:
        return ""

    reader = csv.reader(lines, delimiter=delimiter)
    rows = list(reader)
    if not rows:
        return ""

    headers = [h.strip() for h in rows[0]]
    formatted_rows: list[str] = []
    for idx, row in enumerate(rows[1:], start=1):
        items = []
        for h_idx, col in enumerate(row):
            h = headers[h_idx] if h_idx < len(headers) else f"col_{h_idx}"
            if col.strip():
                items.append(f"{h}: {col.strip()}")
        if items:
            formatted_rows.append(f"[Row {idx}] " + " | ".join(items))

    return "\n".join(formatted_rows)


def _read_html(path: Path) -> str:
    """Extrai texto visível de arquivos HTML limpando scripts e estilos."""
    from html.parser import HTMLParser

    class HTMLTextExtractor(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.result: list[str] = []
            self._ignore = False

        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            if tag in {"script", "style", "head", "meta", "noscript"}:
                self._ignore = True

        def handle_endtag(self, tag: str) -> None:
            if tag in {"script", "style", "head", "meta", "noscript"}:
                self._ignore = False
            elif tag in {"p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "br"}:
                self.result.append("\n")

        def handle_data(self, data: str) -> None:
            if not self._ignore and data.strip():
                self.result.append(data.strip() + " ")

    raw = path.read_text(encoding="utf-8", errors="replace")
    parser = HTMLTextExtractor()
    parser.feed(raw)
    return "".join(parser.result)


def _read_json(path: Path) -> str:
    """Formata arquivos JSON/JSONL de forma estruturada e legível para busca textual."""
    content = path.read_text(encoding="utf-8", errors="replace").strip()
    if not content:
        return ""

    if path.suffix.lower() == ".jsonl":
        lines_out: list[str] = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                lines_out.append(json.dumps(obj, ensure_ascii=False))
            except Exception:  # noqa: BLE001
                lines_out.append(line)
        return "\n".join(lines_out)

    try:
        obj = json.loads(content)
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        return content


def read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf(path)
    if suffix == ".docx":
        return _read_docx(path)
    if suffix in {".csv", ".tsv"}:
        return _read_csv_tsv(path)
    if suffix in {".html", ".htm"}:
        return _read_html(path)
    if suffix in {".json", ".jsonl"}:
        return _read_json(path)
    return path.read_text(encoding="utf-8", errors="replace")


def ingest_directory(
    corpus_dir: str,
    settings: Settings,
    vector_store: VectorStore,
    lexical_store: LexicalStore,
    embedder: EmbeddingProvider,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> dict[str, int | dict[str, int]]:
    """Indexa todos os documentos suportados do diretório nos dois índices."""
    root = Path(corpus_dir)
    if not root.exists():
        raise FileNotFoundError(f"Diretório de corpus não encontrado: {corpus_dir}")

    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap
    per_doc: dict[str, int] = {}
    total_chunks: list[DocumentChunk] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTS:
            continue
        text = read_document(path)
        chunks = chunk_text(path.name, text, chunk_size=chunk_size, overlap=overlap)
        total_chunks.extend(chunks)
        per_doc[path.name] = len(chunks)

    # Sincronização incremental: embed somente chunks novos, poda órfãos.
    sync = vector_store.sync_chunks(total_chunks)
    stored = vector_store.all_chunks()
    lexical_store.rebuild(stored)

    return {
        "documents": len(per_doc),
        "chunks": len(total_chunks),
        "added": sync["added"],
        "deleted": sync["deleted"],
        "unchanged": sync["unchanged"],
        "indexed": len(stored),
        "per_doc": per_doc,
    }
