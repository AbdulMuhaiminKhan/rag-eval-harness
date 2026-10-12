"""Minimal per-query tracing: named, nested spans with wall-clock durations.

A Trace records where a query's time goes (dense search, BM25, fusion, reranking, generation,
judging). Eval runs store each question's spans in per_question.jsonl and summarise per-stage
p50/p95 latency, so a slow config can be explained, not just measured.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class Span:
    name: str
    start_ms: float
    duration_ms: float = 0.0
    attrs: dict = field(default_factory=dict)


class Trace:
    def __init__(self) -> None:
        self._t0 = time.perf_counter()
        self.spans: list[Span] = []
        self._stack: list[str] = []

    @contextmanager
    def span(self, name: str, **attrs) -> Iterator[Span]:
        full = "/".join([*self._stack, name])
        s = Span(full, (time.perf_counter() - self._t0) * 1000, attrs=attrs)
        self._stack.append(name)
        t = time.perf_counter()
        try:
            yield s
        finally:
            s.duration_ms = (time.perf_counter() - t) * 1000
            self._stack.pop()
            self.spans.append(s)

    def to_dict(self) -> list[dict]:
        return [{"name": s.name, "start_ms": round(s.start_ms, 2),
                 "duration_ms": round(s.duration_ms, 2), **({"attrs": s.attrs} if s.attrs else {})}
                for s in sorted(self.spans, key=lambda s: s.start_ms)]


class NullTrace(Trace):
    """A trace that records nothing, so callers never need to check for None."""

    @contextmanager
    def span(self, name: str, **attrs) -> Iterator[Span]:
        yield Span(name, 0.0)


NULL_TRACE = NullTrace()
