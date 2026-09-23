"""Testes do llm-as-judge — parsing e agregação, sem Ollama (fakes)."""

from __future__ import annotations

from hybrid_rag_mcp.judge import JudgeReport, format_judge_report, parse_verdict, run_eval


class _Resp:
    def __init__(self, text: str) -> None:
        self.text = text


def test_parse_verdict_nota_e_justificativa() -> None:
    score, just = parse_verdict("NOTA: 2\nJUSTIFICATIVA: menciona a porta 8443.")
    assert score == 2
    assert just == "menciona a porta 8443."


def test_parse_verdict_case_insensitive_com_espacos() -> None:
    score, _ = parse_verdict("  nota : 1  \njustificativa: parcial.")
    assert score == 1


def test_parse_verdict_sem_nota_nao_vira_acerto() -> None:
    score, _ = parse_verdict("A resposta parece boa, recomendo aprovar.")
    assert score is None


def test_parse_verdict_nota_fora_da_faixa_rejeitada() -> None:
    score, _ = parse_verdict("NOTA: 5\nJUSTIFICATIVA: excelente.")
    assert score is None


def test_run_eval_agrega_e_isola_falhas() -> None:
    items = [
        {"id": "a1", "question": "porta?", "expected": "8443"},
        {"id": "a2", "question": "manutenção?", "expected": "02h às 04h"},
        {"id": "a3", "question": "quebra?", "expected": "x"},
    ]

    def ask_fn(q: str):
        if q == "quebra?":
            raise RuntimeError("LLM fora do ar")
        return _Resp(f"resposta para {q}")

    def judge_fn(system: str, user: str):
        return _Resp("NOTA: 2\nJUSTIFICATIVA: ok." if "porta?" in user else "texto sem nota")

    report = run_eval(items, ask_fn, judge_fn)
    assert isinstance(report, JudgeReport)
    assert len(report.verdicts) == 3
    assert report.count(2) == 1
    assert report.count(None) == 2  # sem veredito + erro de geração
    assert report.mean_score() == 2.0  # média só sobre os avaliados

    text = format_judge_report(report)
    assert "média: 2.00" in text
    assert "SEM VEREDITO" in text
    assert "erro: LLM fora do ar" in text
