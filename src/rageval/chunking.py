"""Chunkers. Every chunk keeps (doc_id, start, end) character offsets into the cleaned document,
which is how retrieval results are matched against gold evidence spans regardless of strategy.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from .ingest import Document


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    start: int
    end: int
    text: str


def _make(doc: Document, start: int, end: int, i: int) -> Chunk:
    return Chunk(f"{doc.doc_id}#{i}", doc.doc_id, start, end, doc.text[start:end])


def fixed_chunks(doc: Document, size: int = 1000, overlap: int = 150) -> list[Chunk]:
    """Sliding character windows, with boundaries nudged back to the nearest whitespace."""
    text, n = doc.text, len(doc.text)
    chunks, start, i = [], 0, 0
    while start < n:
        end = min(start + size, n)
        if end < n:
            ws = text.rfind("\n", start + size // 2, end)
            if ws == -1:
                ws = text.rfind(" ", start + size // 2, end)
            if ws != -1:
                end = ws
        chunks.append(_make(doc, start, end, i))
        i += 1
        if end >= n:
            break
        nxt = max(end - overlap, start + 1)
        sp = text.find(" ", nxt, end)  # avoid starting mid-word
        start = sp + 1 if sp != -1 else nxt
    return chunks


_SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9`])")


def _units(text: str, max_unit: int) -> list[tuple[int, int]]:
    """Split into paragraph units, breaking long paragraphs on sentence boundaries."""
    units = []
    for m in re.finditer(r"[^\n](?:.|\n(?!\n))*", text):
        s, e = m.start(), m.end()
        if e - s <= max_unit:
            units.append((s, e))
            continue
        cur = s
        for sm in _SENT_END.finditer(text, s, e):
            if sm.start() - cur >= max_unit // 2:
                units.append((cur, sm.start()))
                cur = sm.end()
        units.append((cur, e))
    return units


def semantic_chunks(
    doc: Document,
    embed: Callable[[list[str]], np.ndarray],
    breakpoint_percentile: float = 85.0,
    min_size: int = 300,
    max_size: int = 1500,
) -> list[Chunk]:
    """Group adjacent paragraphs, starting a new chunk where the embedding distance between
    neighbours is in the top (100 - percentile)% for this document, or the size cap is hit."""
    units = _units(doc.text, max_unit=max_size // 2)
    if len(units) <= 1:
        return [_make(doc, 0, len(doc.text), 0)] if doc.text else []
    vecs = embed([doc.text[s:e] for s, e in units])
    dist = 1.0 - np.sum(vecs[:-1] * vecs[1:], axis=1)  # vectors are L2-normalised
    threshold = np.percentile(dist, breakpoint_percentile)

    chunks, cur_start, i = [], units[0][0], 0
    for j in range(1, len(units)):
        prev_end, (s, e) = units[j - 1][1], units[j]
        size_so_far = prev_end - cur_start
        if (dist[j - 1] >= threshold and size_so_far >= min_size) or e - cur_start > max_size:
            chunks.append(_make(doc, cur_start, prev_end, i))
            i += 1
            cur_start = s
    chunks.append(_make(doc, cur_start, units[-1][1], i))
    return chunks


def chunk_corpus(docs: Iterable[Document], strategy: str, params: dict,
                 embed: Callable[[list[str]], np.ndarray] | None = None) -> list[Chunk]:
    out: list[Chunk] = []
    for d in docs:
        if strategy == "fixed":
            out.extend(fixed_chunks(d, **params))
        elif strategy == "semantic":
            if embed is None:
                raise ValueError("semantic chunking needs an embedding function")
            out.extend(semantic_chunks(d, embed, **params))
        else:
            raise ValueError(f"unknown chunking strategy: {strategy}")
    return out
