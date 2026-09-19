"""Baixa documentos reais de acesso aberto (Gutenberg/domínio público) e gera um
dataset de avaliação auto-supervisionado a partir do próprio texto baixado.

Uso:
    python tools/fetch_corpus.py                  # baixa catálogo + regrava eval/dataset.real.jsonl
    python tools/fetch_corpus.py --max-bytes 200000 --queries-per-doc 4
    python tools/fetch_corpus.py --dry-run        # só mostra o plano, sem baixar

O dataset gerado é "extractivo": cada consulta é um trecho real do documento e o
documento esperado é a fonte exata desse trecho. Assim o ground-truth é verificável
sem curadoria manual.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx

CORPUS_DIR = Path("examples/corpus")
EVAL_OUT = Path("eval/dataset.real.jsonl")

CATALOG: list[tuple[str, int]] = [
    # (nome do arquivo, id no Project Gutenberg) — todas de domínio público
    ("gutenberg-black-monk.txt", 55307),  # Chekhov, "The Black Monk" (histórias), EN
    ("gutenberg-dom-casmurro.txt", 55752),  # Machado de Assis, PT
    ("gutenberg-cortico.txt", 23291),  # Aluísio Azevedo, PT
    ("gutenberg-primo-basilio.txt", 24719),  # Eça de Queirós, PT
    ("gutenberg-pride-prejudice.txt", 1342),  # Jane Austen, EN
    ("gutenberg-sherlock.txt", 1661),  # Conan Doyle, EN
]

GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"


def fetch_book(slug: str, book_id: int, max_bytes: int, client: httpx.Client) -> bytes | None:
    url = GUTENBERG_URL.format(id=book_id)
    resp = client.get(url, follow_redirects=True, timeout=60)
    if resp.status_code != 200:
        print(f"  ! {slug}: HTTP {resp.status_code} — pulando")
        return None
    resp.raise_for_status()
    return resp.content[:max_bytes]


def write_doc(slug: str, content: bytes) -> None:
    # Remove todo o boilerplate do Gutenberg (cabeçalho e rodapé padrão).
    text = content.decode("utf-8", errors="replace")
    text = strip_gutenberg(text)
    (CORPUS_DIR / slug).write_text(text.strip() + "\n", encoding="utf-8")


def strip_gutenberg(text: str) -> str:
    upper = text.upper()
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG",
        "*** START OF THIS PROJECT GUTENBERG",
    ]
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG",
        "*** END OF THIS PROJECT GUTENBERG",
    ]
    body = text
    for marker in start_markers:
        idx = upper.find(marker)
        if idx != -1:
            newline = body.find("\n", idx)
            body = body[newline + 1 :] if newline != -1 else body[idx:]
            break
    for marker in end_markers:
        idx = body.upper().find(marker)
        if idx != -1:
            body = body[:idx]
            break
    return body


_BOILERPLATE = (
    "project gutenberg",
    "gutenberg ebook",
    "www.gutenberg",
    "e-books@arol",
    "distributed proofread",
    "transcription",
)


def generate_queries(doc_name: str, text: str, queries_per_doc: int) -> list[dict]:
    """Um trecho distintivo por 'seção' do documento vira consulta; a fonte é o esperado."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return []
    queries: list[dict] = []
    for line in lines[: queries_per_doc * 40]:
        lowered = line.lower()
        if len(line) < 40 or len(line) > 250:
            continue
        if any(ban in lowered for ban in _BOILERPLATE):
            continue
        if line.isupper():
            continue
        queries.append(
            {"id": f"real-{doc_name}-{len(queries)}", "query": line[:160], "relevant": [doc_name]}
        )
        if len(queries) >= queries_per_doc:
            break
    return queries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-bytes", type=int, default=200_000)
    parser.add_argument("--queries-per-doc", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_OUT.parent.mkdir(parents=True, exist_ok=True)

    print(f"Plano ({len(CATALOG)} documentos, {args.max_bytes} bytes/doc):")
    for slug, _ in CATALOG:
        print(f"  - {slug}")
    if args.dry_run:
        return

    dataset: list[dict] = []
    with httpx.Client() as client:
        for slug, book_id in CATALOG:
            try:
                content = fetch_book(slug, book_id, args.max_bytes, client)
                if content is None:
                    continue
                write_doc(slug, content)
                text = (CORPUS_DIR / slug).read_text(encoding="utf-8")
                queries = generate_queries(slug, text, args.queries_per_doc)
                dataset.extend(queries)
                print(f"  ✓ {slug}: {len(text) / 1024:.0f} kB, {len(queries)} consultas geradas")
            except Exception as exc:  # noqa: BLE001 - falha de rede não abate o lote
                print(f"  ! {slug}: {exc} — pulando")

    EVAL_OUT.write_text(
        "".join(json.dumps(q, ensure_ascii=False) + "\n" for q in dataset), encoding="utf-8"
    )
    print(f"\nDataset de avaliação: {len(dataset)} consultas -> {EVAL_OUT}")


if __name__ == "__main__":
    main()
