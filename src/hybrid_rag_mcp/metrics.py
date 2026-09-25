"""Métricas em processo, zero dependências: contadores + latências p50/p95.

Exposição:
  - modo HTTP: `GET /metrics` em formato Prometheus (herda o bearer auth);
  - ambos os transportes: tool MCP `metrics` com resumo humano.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from typing import Any

_MAX_SAMPLES = 1024


def percentiles(xs: list[float]) -> dict[str, float]:
    """p50/p95/p99 por nearest-rank + média; lista vazia → zeros."""
    if not xs:
        return {"n": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    s = sorted(xs)
    n = len(s)

    def pct(p: float) -> float:
        return s[min(n - 1, math.ceil(p / 100 * n) - 1)]

    return {"n": n, "mean": sum(s) / n, "p50": pct(50), "p95": pct(95), "p99": pct(99)}


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, int] = {}
        self._lat: dict[str, deque[float]] = {}

    def inc(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + amount

    def observe(self, name: str, value_ms: float) -> None:
        with self._lock:
            buf = self._lat.setdefault(name, deque(maxlen=_MAX_SAMPLES))
            buf.append(value_ms)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "counters": dict(self._counters),
                "latency_ms": {k: percentiles(list(v)) for k, v in self._lat.items()},
            }

    def render_prometheus(self) -> str:
        snap = self.snapshot()
        lines = []
        for name in sorted(snap["counters"]):
            lines += [f"# TYPE rag_{name} counter", f"rag_{name} {snap['counters'][name]}"]
        for op in sorted(snap["latency_ms"]):
            stats = snap["latency_ms"][op]
            lines.append("# TYPE rag_latency_ms summary")
            for q, key in (("0.5", "p50"), ("0.95", "p95")):
                lines.append(f'rag_latency_ms{{op="{op}",quantile="{q}"}} {stats[key]}')
            lines.append(f'rag_latency_ms_sum{{op="{op}"}} {stats["mean"] * stats["n"]}')
            lines.append(f'rag_latency_ms_count{{op="{op}"}} {stats["n"]}')
        return "\n".join(lines) + "\n"

    def render_text(self) -> str:
        snap = self.snapshot()
        lines = ["Métricas do servidor (desde o boot):"]
        for name in sorted(snap["counters"]):
            lines.append(f"- {name}: {snap['counters'][name]}")
        for op in sorted(snap["latency_ms"]):
            stats = snap["latency_ms"][op]
            lines.append(
                f"- {op}: n={stats['n']} média={stats['mean']:.1f}ms "
                f"p50={stats['p50']:.1f}ms p95={stats['p95']:.1f}ms"
            )
        return "\n".join(lines)
