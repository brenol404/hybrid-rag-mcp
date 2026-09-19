from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    doc_name: str
    content: str
    index: int


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    doc_name: str
    content: str
    score: float
    strategy: Literal["vector", "lexical", "hybrid"] = "hybrid"


@dataclass
class TraceStep:
    """Rastreabilidade de uma etapa do pipeline (audit log)."""

    step: str
    detail: str = ""
    ok: bool = True


@dataclass
class AskResult:
    answer: str
    sources: list[SearchHit] = field(default_factory=list)
    provider: str = ""
    model: str = ""
    trace: list[TraceStep] = field(default_factory=list)
    iterations: int = 1
