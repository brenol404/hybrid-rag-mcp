"""Testes do compressor de contexto estilo-Caveman — puros, sem Ollama."""

from __future__ import annotations

from hybrid_rag_mcp.models import SearchHit
from hybrid_rag_mcp.optimize.compress import make_compressor, token_savings
from hybrid_rag_mcp.providers.base import LLMResponse
from hybrid_rag_mcp.rag.agent import run_agent


def test_level_0_returns_none() -> None:
    assert make_compressor(0) is None
    assert make_compressor(-1) is None


def test_light_keeps_facts_numbers_and_names() -> None:
    compress = make_compressor(1)
    assert compress is not None
    out = compress("O servidor usa Redis 7 e RTX 4090, quando necessário.")
    assert out == "O servidor usa Redis 7 RTX 4090, necessário."
    assert "4090" in out and "Redis" in out


def test_aggressive_strips_determiners_and_auxiliaries() -> None:
    compress = make_compressor(2)
    assert compress is not None
    out = compress("A janela de manutenção é das 02h às 04h.")
    assert "A" not in out and "é" not in out
    assert "02h" in out and "04h" in out


def test_negation_is_never_stripped() -> None:
    compress = make_compressor(2)
    assert compress is not None
    out = compress("O backup não é incremental.")
    assert "não" in out


def test_never_returns_empty() -> None:
    compress = make_compressor(2)
    assert compress is not None
    out = compress("o e um")
    assert out  # preserva o original em vez de esvaziar


def test_token_savings_estimates_reduction() -> None:
    compress = make_compressor(2)
    assert compress is not None
    text = "O cache usa Redis e Qdrant, porque ambos são locais."
    out = compress(text)
    assert 0.0 < token_savings(text, out) < 1.0


def test_agent_compresses_prompt_but_keeps_original_sources() -> None:
    def shouting(text: str) -> str:
        return text.upper()

    result = run_agent(
        "pergunta",
        retrieve=lambda q, k: [
            SearchHit("a", "doc.txt", "os servidores ficam em São Paulo", 0.9, "vector")
        ],
        generate=lambda system, user: LLMResponse("resposta [doc.txt].", "ollama", "qwen3"),
        max_iterations=2,
        compress=shouting,
    )
    assert result.answer == "resposta [doc.txt]."
    # fontes preservam o texto ORIGINAL (compressão é só na cópia do prompt)
    assert "ficam em São Paulo" in result.sources[0].content
