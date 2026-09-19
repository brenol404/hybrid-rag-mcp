"""Testes do loop de agente — stubs puros, sem Ollama."""

from __future__ import annotations

from hybrid_rag_mcp.models import SearchHit
from hybrid_rag_mcp.providers.base import LLMResponse
from hybrid_rag_mcp.rag.agent import extract_follow_up, run_agent


def _hits(*names: str) -> list[SearchHit]:
    return [SearchHit(n, n, f"conteúdo de {n}", 0.9, "vector") for n in names]


def test_extract_follow_up() -> None:
    assert extract_follow_up("\n[MORE_CONTEXT] que horas é o backup?\n") == "que horas é o backup?"
    assert extract_follow_up("resposta normal [x].") is None
    assert extract_follow_up("[more_context] busca extra") == "busca extra"


def test_agente_responde_em_uma_iteracao() -> None:
    result = run_agent(
        "pergunta",
        retrieve=lambda q, k: _hits("a"),
        generate=lambda system, user: LLMResponse("resposta com [a].", "ollama", "qwen3"),
        max_iterations=2,
    )
    assert result.iterations == 1
    assert result.answer == "resposta com [a]."
    assert [s.chunk_id for s in result.sources] == ["a"]
    steps = [t.step for t in result.trace]
    assert "retrieval" in steps and "generation" in steps


def test_agente_busca_mais_quando_falta_contexto() -> None:
    answers = iter(
        [
            LLMResponse("[MORE_CONTEXT] qual a janela de manutenção?", "ollama", "qwen3"),
            LLMResponse("A janela é 02h-04h [b].", "ollama", "qwen3"),
        ]
    )

    def retrieve(query: str, k: int) -> list[SearchHit]:
        return _hits("b") if "manutenção" in query else _hits("a")

    result = run_agent(
        "pergunta inicial",
        retrieve=retrieve,
        generate=lambda system, user: next(answers),
        max_iterations=2,
    )
    assert result.iterations == 2
    # memória: fontes da 1ª iteração não re-entram, e a 2ª adiciona novas
    assert [s.chunk_id for s in result.sources] == ["a", "b"]
    steps = [t.step for t in result.trace]
    assert steps.count("retrieval") == 2
    assert "follow_up" in steps


def test_agente_cap_no_limite_de_iteracoes() -> None:
    def retrieve(q: str, k: int) -> list[SearchHit]:
        return _hits("bm25_extra")

    result = run_agent(
        "pergunta",
        retrieve=retrieve,
        generate=lambda system, user: LLMResponse("[MORE_CONTEXT] ainda falta", "ollama", "qwen3"),
        max_iterations=2,
    )
    assert result.iterations == 2
    assert any(t.step == "generation" and not t.ok for t in result.trace)
