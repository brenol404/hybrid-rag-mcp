"""Avaliação da qualidade das RESPOSTAS (não só do retrieval): llm-as-judge.

O gate de retrieval (`recall@1`) prova que o trecho certo sobe; este prova que
a resposta final está correta. Cada item de `eval/answers.jsonl` tem pergunta +
resposta esperada: o engine responde via `ask()` e o próprio LLM (Ollama) dá
NOTA 0/1/2 com justificativa de uma frase.

Uso:
    python -m hybrid_rag_mcp.judge                    # 12 perguntas sobre o corpus
    python -m hybrid_rag_mcp.judge --fail-under 1.5   # falha se média < 1.5 (0..2)

Precisa do Ollama de pé (embeddings + geração + juiz). Itens sem NOTA
parseável contam como "sem veredito" — nunca como acerto silencioso.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field

from .config import get_settings
from .rag.engine import RAGEngine

JUDGE_SYSTEM = (
    "Você é um avaliador rigoroso de respostas de um sistema RAG. "
    "Compare a RESPOSTA com a RESPOSTA ESPERADA. Ignore citações de fonte e "
    "formatação; avalie só o conteúdo factual. Responda EXATAMENTE neste formato:\n"
    "NOTA: <0, 1 ou 2>\n"
    "JUSTIFICATIVA: <uma frase>\n"
    "0 = errada ou não responde; 1 = parcialmente correta; 2 = correta e completa."
)

_NOTA = re.compile(r"^\s*NOTA\s*:\s*([012])\s*$", re.IGNORECASE | re.MULTILINE)


@dataclass(frozen=True)
class ItemVerdict:
    query_id: str
    question: str
    expected: str
    answer: str
    score: int | None
    justification: str


@dataclass
class JudgeReport:
    verdicts: list[ItemVerdict] = field(default_factory=list)

    def scored(self) -> list[ItemVerdict]:
        return [v for v in self.verdicts if v.score is not None]

    def mean_score(self) -> float:
        s = self.scored()
        return sum(v.score for v in s) / len(s) if s else 0.0

    def count(self, score: int | None) -> int:
        return sum(1 for v in self.verdicts if v.score == score)


def load_answers(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def parse_verdict(text: str) -> tuple[int | None, str]:
    """Extrai (nota, justificativa) da saída do juiz; sem NOTA → sem veredito."""
    m = _NOTA.search(text)
    justification = ""
    for line in text.splitlines():
        if line.strip().upper().startswith("JUSTIFICATIVA"):
            justification = line.split(":", 1)[1].strip() if ":" in line else ""
            break
    if m is None:
        return None, justification or text.strip()[:160]
    return int(m.group(1)), justification


def judge_one(question: str, expected: str, answer: str, judge_fn) -> tuple[int | None, str]:
    user = f"PERGUNTA: {question}\nRESPOSTA ESPERADA: {expected}\nRESPOSTA: {answer}"
    resp = judge_fn(JUDGE_SYSTEM, user)
    text = resp.text if hasattr(resp, "text") else str(resp)
    return parse_verdict(text)


def run_eval(items: list[dict], ask_fn, judge_fn) -> JudgeReport:
    report = JudgeReport()
    for item in items:
        try:
            answer = ask_fn(item["question"])
            answer_text = answer.answer if hasattr(answer, "answer") else str(answer)
        except Exception as exc:  # noqa: BLE001 - erro de geração vira item sem veredito
            report.verdicts.append(
                ItemVerdict(
                    item["id"], item["question"], item["expected"], "", None, f"erro: {exc}"
                )
            )
            continue
        score, justification = judge_one(item["question"], item["expected"], answer_text, judge_fn)
        report.verdicts.append(
            ItemVerdict(
                item["id"], item["question"], item["expected"], answer_text, score, justification
            )
        )
    return report


def format_judge_report(report: JudgeReport) -> str:
    lines = [
        f"Judge · {len(report.verdicts)} perguntas, {len(report.scored())} com veredito",
        (
            f"média: {report.mean_score():.2f} (0..2)  |  2s: {report.count(2)}  "
            f"1s: {report.count(1)}  0s: {report.count(0)}  sem veredito: {report.count(None)}"
        ),
        "",
    ]
    for v in report.verdicts:
        mark = f"NOTA {v.score}" if v.score is not None else "SEM VEREDITO"
        lines.append(f"{v.query_id} [{mark}] {v.question}")
        lines.append(f"    esperado: {v.expected}")
        lines.append(f"    justificativa: {v.justification}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", default="eval/answers.jsonl")
    parser.add_argument("--corpus", default=None)
    parser.add_argument("--fail-under", type=float, default=None)
    args = parser.parse_args()

    settings = get_settings()
    if args.corpus:
        settings.corpus_dir = args.corpus

    engine = RAGEngine(settings)
    engine.ingest(settings.corpus_dir)
    items = load_answers(args.answers)
    print(f"Avaliando {len(items)} respostas...\n")
    report = run_eval(
        items,
        ask_fn=lambda q: engine.ask(q),
        judge_fn=lambda system, user: engine._llm.complete(system, user),
    )
    print(format_judge_report(report))
    engine.close()

    mean = report.mean_score()
    if args.fail_under is not None and mean < args.fail_under:
        raise SystemExit(f"média {mean:.2f} abaixo do limite {args.fail_under}")


if __name__ == "__main__":
    main()
