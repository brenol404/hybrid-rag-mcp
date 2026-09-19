"""Avaliação do pipeline de retrieval: recall@k e nDCG@k sobre dataset de ground-truth.

Uso:
    python -m hybrid_rag_mcp.eval                     # usa eval/dataset.jsonl + corpus padrão
    python -m hybrid_rag_mcp.eval --dataset path.jsonl --corpus dir/
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field

from .config import get_settings
from .rag.engine import RAGEngine


@dataclass(frozen=True)
class QueryOutcome:
    query_id: str
    query: str
    relevant: set[str]
    hits: list[str]
    recall_at: dict[int, float]
    ndcg_at: dict[int, float]


@dataclass
class EvalReport:
    outcomes: list[QueryOutcome] = field(default_factory=list)
    ks: tuple[int, ...] = (1, 3, 5)

    @property
    def n_queries(self) -> int:
        return len(self.outcomes)

    def mean_recall(self, k: int) -> float:
        vals = [o.recall_at[k] for o in self.outcomes if k in o.recall_at]
        return sum(vals) / len(vals) if vals else 0.0

    def mean_ndcg(self, k: int) -> float:
        vals = [o.ndcg_at[k] for o in self.outcomes if k in o.ndcg_at]
        return sum(vals) / len(vals) if vals else 0.0


def load_dataset(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def relevance(hit_doc: str, relevant: set[str]) -> float:
    return float(hit_doc in relevant)


def ndcg_at_k(rel: list[float], k: int) -> float:
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel[:k]))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, sum(1 for r in rel if r))))
    return dcg / idcg if idcg else 0.0


def recall_at_k(rel: list[float], relevant_total: int, k: int) -> float:
    if relevant_total == 0:
        return 0.0
    return min(sum(rel[:k]) / relevant_total, 1.0)


def evaluate(
    engine: RAGEngine,
    dataset: list[dict],
    ks: tuple[int, ...] = (1, 3, 5),
) -> EvalReport:
    report = EvalReport(ks=ks)
    for item in dataset:
        relevant = set(item["relevant"])
        hits = [h.doc_name for h in engine.search(item["query"], top_k=max(ks))]
        rel = [relevance(doc, relevant) for doc in hits]
        report.outcomes.append(
            QueryOutcome(
                query_id=item["id"],
                query=item["query"],
                relevant=relevant,
                hits=hits[: max(ks)],
                recall_at={k: recall_at_k(rel, len(relevant), k) for k in ks},
                ndcg_at={k: ndcg_at_k(rel, k) for k in ks},
            )
        )
    return report


def format_report(report: EvalReport) -> str:
    lines = [
        f"Eval · {report.n_queries} queries",
        "",
        f"{'k':<4}{'recall@k':>12}{'nDCG@k':>12}",
        "-" * 28,
    ]
    for k in report.ks:
        r = report.mean_recall(k)
        n = report.mean_ndcg(k)
        lines.append(f"{k:<4}{r:>12.3f}{n:>12.3f}")
    lines.append("")
    for o in report.outcomes:
        recall1 = o.recall_at[report.ks[0]]
        flag = "  OK " if recall1 > 0 else "  FALHOU"
        found = ", ".join(f"{d}" for d in o.hits[:3])
        lines.append(f"{o.query_id}{flag} esperado={sorted(o.relevant)} top={found}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="eval/dataset.jsonl")
    parser.add_argument("--corpus", default=None)
    parser.add_argument("--ks", default="1,3,5", help="valores de k, separados por vírgula")
    args = parser.parse_args()

    settings = get_settings()
    if args.corpus:
        settings.corpus_dir = args.corpus

    engine = RAGEngine(settings)
    engine.ingest(settings.corpus_dir)
    ks = tuple(int(k) for k in args.ks.split(","))

    report = evaluate(engine, load_dataset(args.dataset), ks)
    print(format_report(report))


if __name__ == "__main__":
    main()
