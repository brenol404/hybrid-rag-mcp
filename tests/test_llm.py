"""Testes da cascata de LLMs — parse da chain e ordem de fallback, sem rede."""

from __future__ import annotations

import pytest

from hybrid_rag_mcp.config import Settings
from hybrid_rag_mcp.providers.base import LLMProvider, LLMResponse
from hybrid_rag_mcp.providers.llm import CloudLLM, FallbackLLM, parse_llm_chain


class _Failer(LLMProvider):
    provider_name = "falho"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, system: str, user: str) -> LLMResponse:
        self.calls += 1
        raise ConnectionError("provedor caiu")

    def complete_stream(self, system: str, user: str):
        raise ConnectionError("provedor caiu")
        yield  # noqa: unreachable — só para virar gerador


class _Ok(LLMProvider):
    provider_name = "certo"

    def __init__(self, text: str = "resposta") -> None:
        self.text = text
        self.calls = 0

    def complete(self, system: str, user: str) -> LLMResponse:
        self.calls += 1
        return LLMResponse(self.text, "certo", "m")

    def complete_stream(self, system: str, user: str):
        self.calls += 1
        yield LLMResponse(self.text, "certo", "m")


def test_parse_chain_vazia_e_trio() -> None:
    assert parse_llm_chain(Settings()) == []
    single = parse_llm_chain(
        Settings(cloud_base_url="https://a.com", cloud_api_key="k", cloud_model="m")
    )
    assert single == [{"base_url": "https://a.com", "api_key": "k", "model": "m"}]


def test_parse_chain_json_vence_trio() -> None:
    chain = parse_llm_chain(
        Settings(
            cloud_base_url="https://a.com",
            cloud_api_key="k",
            cloud_model="m",
            llm_chain='[{"base_url": "https://b.com", "api_key": "k2", "model": "m2"}]',
        )
    )
    assert chain == [{"base_url": "https://b.com", "api_key": "k2", "model": "m2"}]


def test_parse_chain_erros_claros() -> None:
    with pytest.raises(ValueError, match="JSON inválido"):
        parse_llm_chain(Settings(llm_chain="não-json"))
    with pytest.raises(ValueError, match="lista JSON"):
        parse_llm_chain(Settings(llm_chain='{"base_url": "x"}'))
    with pytest.raises(ValueError, match="LLM_CHAIN\\[1\\]"):
        parse_llm_chain(
            Settings(
                llm_chain='[{"base_url": "a", "api_key": "k", "model": "m"}, {"base_url": "b"}]'
            )
        )


def test_fallback_primeira_ok_vence() -> None:
    fail, ok = _Failer(), _Ok("venceu")
    fb = FallbackLLM(providers=[fail, ok])
    assert fb.complete("s", "u").text == "venceu"
    assert (fail.calls, ok.calls) == (1, 1)


def test_fallback_tudo_falha_agrega_erros() -> None:
    fb = FallbackLLM(providers=[_Failer(), _Failer()])
    with pytest.raises(RuntimeError, match="falho.*falho"):
        fb.complete("s", "u")


def test_fallback_stream_primeiro_a_emitir() -> None:
    fb = FallbackLLM(providers=[_Failer(), _Ok("tok")])
    assert "".join(r.text for r in fb.complete_stream("s", "u")) == "tok"


def test_cloud_params_explicitos(monkeypatch) -> None:
    posts: list[dict] = []

    class _Resp:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"choices": [{"message": {"content": "oi"}}]}

    class _Http:
        def __init__(self, **kwargs) -> None:
            pass

        def post(self, url: str, headers: dict | None = None, json: dict | None = None):
            posts.append({"url": url, "headers": headers, "json": json})
            return _Resp()

        def close(self) -> None:
            pass

    monkeypatch.setattr("httpx.Client", _Http)
    llm = CloudLLM(
        Settings(), base_url="https://nvidia.exemplo/v1", api_key="nv-k", model="modelo-nv"
    )
    resp = llm.complete("s", "u")
    assert resp.text == "oi" and resp.model == "modelo-nv"
    assert posts[0]["url"] == "https://nvidia.exemplo/v1/chat/completions"
    assert posts[0]["headers"]["Authorization"] == "Bearer nv-k"
