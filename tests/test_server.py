"""Validação de entrada das tools MCP — helpers puros, sem engine/Ollama."""

from __future__ import annotations

from pathlib import Path

import pytest

from hybrid_rag_mcp.server import _clamp_top_k, _resolve_corpus_dir


def test_clamp_top_k_limita_abaixo() -> None:
    assert _clamp_top_k(-5) == 1
    assert _clamp_top_k(0) == 1


def test_clamp_top_k_limita_acima() -> None:
    assert _clamp_top_k(10_000) == 50
    assert _clamp_top_k(50) == 50


def test_clamp_top_k_mantem_valores_validos_e_default() -> None:
    assert _clamp_top_k(None) == 5
    assert _clamp_top_k(3) == 3
    assert _clamp_top_k(20) == 20


def test_resolve_corpus_dir_aceita_dir_existente(tmp_path: Path) -> None:
    d = tmp_path / "docs"
    d.mkdir()
    assert _resolve_corpus_dir("", str(d)) == str(d)
    assert _resolve_corpus_dir(str(d), str(d)) == str(d)


def test_resolve_corpus_dir_rejeita_ausente_ou_arquivo(tmp_path: Path) -> None:
    missing = tmp_path / "nao-existe"
    with pytest.raises(ValueError, match="não encontrado"):
        _resolve_corpus_dir("", str(missing))
    file_ = tmp_path / "arquivo.txt"
    file_.write_text("não é diretório", encoding="utf-8")
    with pytest.raises(ValueError, match="não é um diretório"):
        _resolve_corpus_dir(str(file_), str(missing))
