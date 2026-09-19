"""Testes das métricas de avaliação — funções puras, sem Ollama."""

from __future__ import annotations

from hybrid_rag_mcp.eval import ndcg_at_k, recall_at_k


def test_ndcg_perfect_ordering_is_one() -> None:
    assert ndcg_at_k([1.0, 1.0, 1.0], 3) == 1.0


def test_ndcg_relevant_first_beats_relevant_last() -> None:
    good = ndcg_at_k([1.0, 0.0, 1.0], 3)
    bad = ndcg_at_k([1.0, 1.0, 0.0], 3)
    assert good < bad  # nDCG penaliza posição


def test_ndcg_no_relevant_is_zero() -> None:
    assert ndcg_at_k([0.0, 0.0], 2) == 0.0


def test_recall_at_k_caps_at_one() -> None:
    assert recall_at_k([1.0, 1.0, 1.0], 2, 2) == 1.0


def test_recall_partial() -> None:
    assert recall_at_k([1.0, 0.0, 0.0], 2, 1) == 0.5


def test_recall_zero_relevant_is_zero() -> None:
    assert recall_at_k([0.0, 0.0], 0, 2) == 0.0
