"""Eval set schema. Gold labels are evidence *spans* (doc_id + verbatim quote from the cleaned
document), not chunk ids, so the same labels score every chunking strategy."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .ingest import Document

QUESTION_TYPES = ("factoid", "procedural", "conceptual", "comparison", "multi_hop", "unanswerable")


@dataclass
class Evidence:
    """One required piece of evidence. ``alternatives`` are equivalent spans elsewhere in the
    corpus (the same fact stated on another page); covering any one of them counts."""

    doc_id: str
    quote: str
    start: int = -1  # resolved against the corpus at load time
    end: int = -1
    alternatives: list[Evidence] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> Evidence:
        return cls(d["doc_id"], d["quote"],
                   alternatives=[cls(a["doc_id"], a["quote"]) for a in d.get("alternatives", [])])

    def to_dict(self) -> dict:
        d = {"doc_id": self.doc_id, "quote": self.quote}
        if self.alternatives:
            d["alternatives"] = [{"doc_id": a.doc_id, "quote": a.quote} for a in self.alternatives]
        return d

    def spans(self) -> list[Evidence]:
        return [self, *self.alternatives]


@dataclass
class Question:
    qid: str
    question: str
    type: str
    answerable: bool = True
    evidence: list[Evidence] = field(default_factory=list)
    reference_answer: str = ""
    origin: str = "hand"  # hand | llm
    reviewed: bool = False
    reviewer: str = ""  # who checked it: a person's name, or "claude" for a model review pass
    notes: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> Question:
        d = dict(d)
        d["evidence"] = [Evidence.from_dict(e) for e in d.get("evidence", [])]
        return cls(**d)

    def to_dict(self) -> dict:
        return {
            "qid": self.qid, "question": self.question, "type": self.type,
            "answerable": self.answerable,
            "evidence": [e.to_dict() for e in self.evidence],
            "reference_answer": self.reference_answer, "origin": self.origin,
            "reviewed": self.reviewed, "reviewer": self.reviewer, "notes": self.notes,
        }


def find_span(text: str, quote: str) -> tuple[int, int] | None:
    """Locate a quote in text, tolerating whitespace differences. Returns char offsets."""
    i = text.find(quote)
    if i != -1:
        return i, i + len(quote)
    words = quote.split()
    if not words:
        return None
    pattern = r"\s+".join(re.escape(w) for w in words)
    m = re.search(pattern, text)
    return (m.start(), m.end()) if m else None


def load_questions(path: Path, docs: dict[str, Document], reviewed_only: bool = False,
                   strict: bool = True) -> tuple[list[Question], list[str]]:
    """Load questions and resolve evidence spans. Returns (questions, problems)."""
    questions, problems = [], []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            q = Question.from_dict(json.loads(line))
            if reviewed_only and not q.reviewed:
                continue
            if q.type not in QUESTION_TYPES:
                problems.append(f"{q.qid}: unknown type {q.type!r}")
            ok = True
            for e in (span for ev in q.evidence for span in ev.spans()):
                doc = docs.get(e.doc_id)
                span = find_span(doc.text, e.quote) if doc else None
                if span is None:
                    problems.append(f"{q.qid}: evidence not found in {e.doc_id!r}: {e.quote[:60]!r}")
                    ok = False
                else:
                    e.start, e.end = span
            if q.answerable and not q.evidence:
                problems.append(f"{q.qid}: answerable question has no evidence")
                ok = False
            if ok or not strict:
                questions.append(q)
    return questions, problems
