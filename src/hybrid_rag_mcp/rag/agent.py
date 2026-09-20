"""Loop de agente multi-step com memória incremental de fontes.

Fluxo: para cada iteração, recupera novos chunks (excluindo os já usados),
monta contexto cumulativo e pergunta ao LLM. Se o modelo sinalizar contexto
insuficiente com a tag [MORE_CONTEXT] + uma pergunta de pesquisa adicional,
o loop refaz a busca com essa pergunta. Cada passo gera um TraceStep.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator

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
StreamFn = Callable[[str, str], Iterator[LLMResponse]]


def extract_follow_up(text: str) -> str | None:
    m = _MORE_CONTEXT.search(text)
    return m.group(1).strip() if m else None


def format_context(hits: list[SearchHit]) -> str:
    return "\n\n".join(f"[{h.doc_name}] {h.content}" for h in hits)


def _render_context(hits: list[SearchHit], compress: Callable[[str], str] | None) -> str:
    """Formata o contexto; quando `compress` existe, comprime só a cópia do prompt."""
    if compress is None:
        return format_context(hits)
    rendered = [
        SearchHit(
            chunk_id=h.chunk_id,
            doc_name=h.doc_name,
            content=compress(h.content),
            score=h.score,
            strategy=h.strategy,
        )
        for h in hits
    ]
    return format_context(rendered)


def run_agent(
    question: str,
    retrieve: RetrieveFn,
    generate: GenerateFn,
    max_iterations: int = 2,
    top_k: int = 5,
    on_event: Callable[[str], None] | None = None,
    on_tokens: Callable[[str], None] | None = None,
    generate_stream: StreamFn | None = None,
    compress: Callable[[str], str] | None = None,
) -> AskResult:
    used: dict[str, SearchHit] = {}
    trace: list[TraceStep] = [TraceStep("ask", question)]
    current_question = question
    last_answer = ""
    last_provider = last_model = ""

    def emit(message: str) -> None:
        if on_event:
            on_event(message)

    for iteration in range(1, max_iterations + 1):
        emit(f"iteração {iteration}: buscando contexto…")
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
        emit(f"iteração {iteration}: {len(new)} novos trechos, {len(used)} ao total")

        prompt = (
            f"Contexto fornecido até agora:\n{_render_context(list(used.values()), compress)}\n\n"
            f"Pergunta: {current_question}"
        )
        emit(f"iteração {iteration}: gerando resposta…")
        if on_tokens is not None and generate_stream is not None:
            pieces: list[str] = []
            for chunk in generate_stream(SYSTEM_PROMPT, prompt):
                last_provider, last_model = chunk.provider, chunk.model
                pieces.append(chunk.text)
                on_tokens(chunk.text)
            resp = LLMResponse(text="".join(pieces), provider=last_provider, model=last_model)
        else:
            resp = generate(SYSTEM_PROMPT, prompt)
            last_provider, last_model = resp.provider, resp.model
        last_answer = resp.text
        trace.append(
            TraceStep(
                "generation", f"iteração {iteration}: provedor={resp.provider} modelo={resp.model}"
            )
        )

        follow_up = extract_follow_up(resp.text)
        if follow_up is None:
            emit("contexto suficiente — resposta pronta")
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
        emit(f"contexto insuficiente — refazendo busca: {follow_up!r}")
        current_question = follow_up

    trace.append(
        TraceStep("generation", f"limite de {max_iterations} iterações atingido", ok=False)
    )
    emit(f"limite de {max_iterations} iterações atingido")
    return AskResult(
        answer=last_answer,
        sources=list(used.values()),
        provider=last_provider or "?",
        model=last_model or "?",
        trace=trace,
        iterations=max_iterations,
    )
