"""Validação de entrada das tools MCP — helpers puros, sem engine/Ollama."""

from __future__ import annotations

from pathlib import Path

import pytest
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from hybrid_rag_mcp.config import Settings
from hybrid_rag_mcp.server import (
    _BearerAuthMiddleware,
    _build_http_app,
    _clamp_top_k,
    _resolve_corpus_dir,
)


async def _ok(request):
    return PlainTextResponse("ok")


def _authed_app(token: str) -> TestClient:
    app = Starlette(routes=[Route("/", _ok)])
    app.add_middleware(_BearerAuthMiddleware, token=token)
    return TestClient(app, raise_server_exceptions=False)


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


def test_bearer_exige_token_correto() -> None:
    client = _authed_app("segredo")
    assert client.get("/").status_code == 401
    assert client.get("/", headers={"Authorization": "Bearer errado"}).status_code == 401
    resp = client.get("/", headers={"Authorization": "Bearer segredo"})
    assert resp.status_code == 200 and resp.text == "ok"


def test_build_http_app_sem_token_nao_tem_middleware() -> None:
    app = _build_http_app(Settings(mcp_auth_token=""))
    assert not any(m.cls is _BearerAuthMiddleware for m in app.user_middleware)


def test_build_http_app_com_token_pendura_middleware() -> None:
    app = _build_http_app(Settings(mcp_auth_token="segredo"))
    matches = [m for m in app.user_middleware if m.cls is _BearerAuthMiddleware]
    assert len(matches) == 1
    assert matches[0].kwargs.get("token") == "segredo"
