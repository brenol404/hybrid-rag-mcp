from __future__ import annotations

import io
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from hybrid_rag_mcp.rag.chunker import chunk_text
from hybrid_rag_mcp.rag.ingestion import (
    SUPPORTED_EXTS,
    _read_csv_tsv,
    _read_docx,
    _read_html,
    _read_json,
    read_document,
)


def test_supported_extensions() -> None:
    assert {".md", ".txt", ".pdf", ".docx", ".csv", ".tsv", ".html", ".json", ".jsonl"}.issubset(
        SUPPORTED_EXTS
    )


def test_read_csv_and_tsv(tmp_path: Path) -> None:
    csv_file = tmp_path / "data.csv"
    csv_file.write_text(
        "name,role,level\nAlice,Engineer,Senior\nBob,Designer,Mid\n", encoding="utf-8"
    )
    content = _read_csv_tsv(csv_file)
    assert "[Row 1] name: Alice | role: Engineer | level: Senior" in content
    assert "[Row 2] name: Bob | role: Designer | level: Mid" in content

    tsv_file = tmp_path / "data.tsv"
    tsv_file.write_text("service\tport\tstatus\nweb\t8000\tactive\n", encoding="utf-8")
    content_tsv = _read_csv_tsv(tsv_file)
    assert "[Row 1] service: web | port: 8000 | status: active" in content_tsv


def test_read_html(tmp_path: Path) -> None:
    html_file = tmp_path / "page.html"
    html_file.write_text(
        "<html><head><title>Ignored</title><style>.css { color: red; }</style></head>"
        "<body><h1>Manual de Operação</h1><script>alert(1);</script>"
        "<p>O backup diário roda às 03:00.</p></body></html>",
        encoding="utf-8",
    )
    content = _read_html(html_file)
    assert "Manual de Operação" in content
    assert "O backup diário roda às 03:00." in content
    assert "alert" not in content
    assert ".css" not in content


def test_read_json_and_jsonl(tmp_path: Path) -> None:
    json_file = tmp_path / "config.json"
    json_file.write_text('{"app": "hybrid-rag", "version": "0.2.0"}', encoding="utf-8")
    content_json = _read_json(json_file)
    assert '"app": "hybrid-rag"' in content_json

    jsonl_file = tmp_path / "records.jsonl"
    jsonl_file.write_text(
        '{"event": "start", "code": 1}\n{"event": "stop", "code": 0}\n', encoding="utf-8"
    )
    content_jsonl = _read_json(jsonl_file)
    assert '{"event": "start", "code": 1}' in content_jsonl
    assert '{"event": "stop", "code": 0}' in content_jsonl


def test_read_docx_mock(tmp_path: Path) -> None:
    docx_file = tmp_path / "sample.docx"
    ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    root = ET.Element(f"{{{ns}}}document")
    body = ET.SubElement(root, f"{{{ns}}}body")
    p1 = ET.SubElement(body, f"{{{ns}}}p")
    t1 = ET.SubElement(p1, f"{{{ns}}}t")
    t1.text = "Documento Confidencial de Arquitetura."
    p2 = ET.SubElement(body, f"{{{ns}}}p")
    t2 = ET.SubElement(p2, f"{{{ns}}}t")
    t2.text = "O cluster roda em três zonas de disponibilidade."

    xml_bytes = ET.tostring(root, encoding="utf-8")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("word/document.xml", xml_bytes)

    docx_file.write_bytes(buffer.getvalue())
    content = _read_docx(docx_file)
    assert "Documento Confidencial de Arquitetura." in content
    assert "O cluster roda em três zonas de disponibilidade." in content


def test_read_document_dispatcher(tmp_path: Path) -> None:
    txt_file = tmp_path / "note.txt"
    txt_file.write_text("Simples nota de texto.", encoding="utf-8")
    assert read_document(txt_file) == "Simples nota de texto."

    csv_file = tmp_path / "users.csv"
    csv_file.write_text("id,name\n1,Admin\n", encoding="utf-8")
    assert "[Row 1] id: 1 | name: Admin" in read_document(csv_file)


def test_chunker_code_block_preservation() -> None:
    doc = """# Tutorial Python

Aqui está uma função essencial:

```python
def connect():
    host = "127.0.0.1"
    port = 8000
    return f"{host}:{port}"
```

E outra instrução após o código.
"""
    chunks = chunk_text("tutorial.md", doc, chunk_size=500, overlap=50)
    contents = [c.content for c in chunks]
    # O bloco de código ```python ... ``` deve ser mantido íntegro e não fragmentado
    code_found = any("def connect():" in c and 'return f"{host}:{port}"' in c for c in contents)
    assert code_found, "O bloco de código deve ser mantido íntegro no chunk"
