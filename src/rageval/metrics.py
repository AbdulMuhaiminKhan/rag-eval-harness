"""Retrieval metrics over evidence spans, plus bootstrap confidence intervals.

A retrieved chunk *covers* a gold span when they overlap by at least ``min_overlap`` of the
shorter of the two. Each gold span can be credited once, so a chunker that slices one passage
into many tiny chunks does not inflate scores.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from .chunking import Chunk
from .dataset import Evidence


def _covers_span(chunk: Chunk, ev: Evidence, min_overlap: float) -> bool:
    if chunk.doc_id != ev.doc_id:
        return False
    overlap = min(chunk.end, ev.end) - max(chunk.start, ev.start)
    if overlap <= 0:
        return False
    return overlap / min(chunk.end - chunk.start, ev.end - ev.start) >= min_overlap


def covers(chunk: Chunk, ev: Evidence, min_overlap: float = 0.5) -> bool:
    """True if the chunk covers the gold span or any of its equivalent alternatives."""
    return any(_covers_span(chunk, s, min_overlap) for s in ev.spans())


def gains(ranked: Sequence[Chunk], evidence: Sequence[Evidence], min_overlap: float = 0.5
          ) -> list[int]:
    """Binary gain per rank: 1 if the chunk covers a gold span not already credited."""
    credited: set[int] = set()
    out = []
    for c in ranked:
        g = 0
        for j, ev in enumerate(evidence):
            if j not in credited and covers(c, ev, min_overlap):
                credited.add(j)
                g = 1
                break
        out.append(g)
    return out


def recall_at_k(g: Sequence[int], n_gold: int, k: int) -> float:
    return sum(g[:k]) / n_gold if n_gold else 0.0


def mrr_at_k(g: Sequence[int], k: int) -> float:
    for i, x in enumerate(g[:k], start=1):
        if x:
            return 1.0 / i
    return 0.0


def ndcg_at_k(g: Sequence[int], n_gold: int, k: int) -> float:
    dcg = sum(x / math.log2(i + 1) for i, x in enumerate(g[:k], start=1))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(k, n_gold) + 1))
    return dcg / idcg if idcg else 0.0


def doc_recall_at_k(ranked: Sequence[Chunk], evidence: Sequence[Evidence], k: int) -> float:
    """Fraction of evidence items with at least one acceptable source page in the top k."""
    got = {c.doc_id for c in ranked[:k]}
    hit = [any(s.doc_id in got for s in e.spans()) for e in evidence]
    return sum(hit) / len(hit) if hit else 0.0


def retrieval_scores(ranked: Sequence[Chunk], evidence: Sequence[Evidence],
                     ks: Sequence[int] = (1, 3, 5, 10), min_overlap: float = 0.5
                     ) -> dict[str, float]:
    g = gains(ranked, evidence, min_overlap)
    n = len(evidence)
    out: dict[str, float] = {}
    for k in ks:
        out[f"recall@{k}"] = recall_at_k(g, n, k)
        out[f"ndcg@{k}"] = ndcg_at_k(g, n, k)
        out[f"doc_recall@{k}"] = doc_recall_at_k(ranked, evidence, k)
    out[f"mrr@{max(ks)}"] = mrr_at_k(g, max(ks))
    return out


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float, float]:
    """Percentile bootstrap of the mean over questions. Returns (mean, low, high)."""
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    boots = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def paired_bootstrap_diff(a: Sequence[float], b: Sequence[float], n_boot: int = 2000,
                          alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """CI on mean(b) - mean(a) for per-question scores of two systems on the same questions."""
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    return bootstrap_ci(d, n_boot, alpha, seed)
