"""Loop de agente multi-step com memória incremental de fontes.

Fluxo: para cada iteração, recupera novos chunks (excluindo os já usados),
monta contexto cumulativo e pergunta ao LLM. Se o modelo sinalizar contexto
insuficiente com a tag [MORE_CONTEXT] + uma pergunta de pesquisa adicional,
o loop refaz a busca com essa pergunta. Cada passo gera um TraceStep.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from ..models import AskResult, SearchHit, TraceStep
from ..providers.base import LLMResponse

SYSTEM_PROMPT = (
    "Você é um assistente técnico que responde usando APENAS o contexto fornecido. "
    "Cite a fonte de cada afirmação entre colchetes, ex.: [manual.pdf]. "
    "Se o contexto for INSUFICIENTE para responder com precisão, responda exatamente "
    "neste formato, sem mais nada:\n"
    "[MORE_CONTEXT] <pergunta de pesquisa adicional, curta e objetiva>\n"
    "Se o contexto for suficiente, responda normalmente, sem usar a tag."
)


_MORE_CONTEXT = re.compile(r"^\s*\[MORE_CONTEXT\]\s*(.+)$", re.IGNORECASE | re.MULTILINE)

RetrieveFn = Callable[[str, int], list[SearchHit]]
GenerateFn = Callable[[str, str], LLMResponse]


def extract_follow_up(text: str) -> str | None:
    m = _MORE_CONTEXT.search(text)
    return m.group(1).strip() if m else None


def format_context(hits: list[SearchHit]) -> str:
    return "\n\n".join(f"[{h.doc_name}] {h.content}" for h in hits)


def run_agent(
    question: str,
    retrieve: RetrieveFn,
    generate: GenerateFn,
    max_iterations: int = 2,
    top_k: int = 5,
) -> AskResult:
    used: dict[str, SearchHit] = {}
    trace: list[TraceStep] = [TraceStep("ask", question)]
    current_question = question
    last_answer = ""

    for iteration in range(1, max_iterations + 1):
        hits = retrieve(current_question, top_k)
        new = [h for h in hits if h.chunk_id not in used]
        for h in new:
            used[h.chunk_id] = h

        trace.append(
            TraceStep(
                "retrieval",
                f"iteração {iteration}: {len(new)} novas fontes de {len(hits)} (total {len(used)})",
            )
        )

        prompt = (
            f"Contexto fornecido até agora:\n{format_context(list(used.values()))}\n\n"
            f"Pergunta: {current_question}"
        )
        resp = generate(SYSTEM_PROMPT, prompt)
        last_answer = resp.text
        trace.append(
            TraceStep(
                "generation", f"iteração {iteration}: provedor={resp.provider} modelo={resp.model}"
            )
        )

        follow_up = extract_follow_up(resp.text)
        if follow_up is None:
            return AskResult(
                answer=resp.text,
                sources=list(used.values()),
                provider=resp.provider,
                model=resp.model,
                trace=trace,
                iterations=iteration,
            )
        trace.append(
            TraceStep("follow_up", f"iteração {iteration}: nova busca → {follow_up!r}", ok=True)
        )
        current_question = follow_up

    trace.append(
        TraceStep("generation", f"limite de {max_iterations} iterações atingido", ok=False)
    )
    return AskResult(
        answer=last_answer,
        sources=list(used.values()),
        provider="?",
        model="?",
        trace=trace,
        iterations=max_iterations,
    )
