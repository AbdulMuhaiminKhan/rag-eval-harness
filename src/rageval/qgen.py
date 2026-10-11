"""Draft eval questions with an LLM, for a human to review before they enter the eval set.

The LLM sees one passage and must return a question plus a verbatim quote from the passage
that answers it. Drafts whose quote cannot be found in the source document are dropped.
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from collections.abc import Callable

from .chunking import fixed_chunks
from .dataset import QUESTION_TYPES, Evidence, Question, find_span
from .generation import LLM
from .ingest import Document
from .retrieval import tokenize

QGEN_PROMPT = """Below is a passage from the Kubernetes documentation (page: {doc_id}).

Write ONE question a Kubernetes user might realistically ask that this passage answers.
The question must make sense without seeing the passage (no "according to the passage"),
and should use the user's own words rather than copying phrases from the text.
Prefer a question of type "{qtype}" if the passage allows it; otherwise pick the best fitting
type from: factoid, procedural, conceptual, comparison.

Passage:
\"\"\"
{passage}
\"\"\"

Respond with JSON only:
{{"question": "...", "type": "...", "answer": "<short reference answer>",
  "quote": "<the exact sentence(s) copied verbatim from the passage that answer it>"}}"""


def _overlap(question: str, quote: str) -> float:
    """Share of the question's content words that appear verbatim in the quote."""
    q, src = set(tokenize(question)), set(tokenize(quote))
    return len(q & src) / len(q) if q else 1.0


def pages_containing(docs: dict[str, Document], quote: str, limit: int = 3) -> int:
    """How many pages contain the quote (stops counting at ``limit``). Boilerplate such as
    kubectl global flags or API list parameters repeats across hundreds of reference pages,
    which makes a single gold page meaningless."""
    probe = " ".join(quote.split())[:120]
    n = 0
    for d in docs.values():
        if probe in " ".join(d.text.split()):
            n += 1
            if n >= limit:
                break
    return n


def _jaccard(a: str, b: str) -> float:
    x, y = set(tokenize(a)), set(tokenize(b))
    return len(x & y) / len(x | y) if x | y else 1.0


def draft_questions(llm: LLM, docs: dict[str, Document], n: int, seed: int = 0,
                    start_index: int = 1, existing: list[str] | None = None,
                    max_overlap: float = 0.75, on_draft: Callable[[Question], None] | None = None
                    ) -> tuple[list[Question], dict[str, int]]:
    """Return (drafts, rejection counts). Drafts that copy the passage's wording, duplicate an
    earlier question, or quote text that is not in the page are rejected. ``on_draft`` is called
    for each accepted draft as soon as it exists, so a long run can be saved incrementally."""
    rng = random.Random(seed)
    # Stratify by top-level section, weighted by sqrt(size) so the huge reference section
    # does not swamp concepts and tasks; tiny sections (home, test pages) are skipped.
    by_section: dict[str, list[Document]] = defaultdict(list)
    for d in docs.values():
        by_section[d.doc_id.split("/")[0]].append(d)
    sections = sorted(s for s, ds in by_section.items() if len(ds) >= 10)
    weights = [math.sqrt(len(by_section[s])) for s in sections]
    target_types = [t for t in QUESTION_TYPES if t not in ("unanswerable", "multi_hop")]

    seen = list(existing or [])
    rejected: dict[str, int] = defaultdict(int)
    out: list[Question] = []
    attempts = 0
    while len(out) < n and attempts < n * 3:
        attempts += 1
        doc = rng.choice(by_section[rng.choices(sections, weights)[0]])
        chunks = [c for c in fixed_chunks(doc, size=1200, overlap=0) if len(c.text) > 400]
        if not chunks:
            rejected["short_page"] += 1
            continue
        passage = rng.choice(chunks).text
        qtype = target_types[attempts % len(target_types)]
        res = llm.complete(QGEN_PROMPT.format(doc_id=doc.doc_id, qtype=qtype, passage=passage),
                           json_mode=True)
        try:
            data = json.loads(res.text)
            question, quote = data["question"].strip(), data["quote"].strip()
        except (json.JSONDecodeError, KeyError, AttributeError):
            rejected["bad_json"] += 1
            continue
        if not question or not (20 <= len(quote) <= 800) or find_span(doc.text, quote) is None:
            rejected["quote_not_in_page"] += 1
            continue
        if _overlap(question, quote) > max_overlap:
            rejected["copies_wording"] += 1
            continue
        if pages_containing(docs, quote) > 1:
            rejected["boilerplate_quote"] += 1
            continue
        if any(_jaccard(question, s) > 0.6 for s in seen):
            rejected["duplicate"] += 1
            continue
        seen.append(question)
        typ = data.get("type") if data.get("type") in QUESTION_TYPES else qtype
        q = Question(
            qid=f"g{start_index + len(out):04d}", question=question, type=typ, answerable=True,
            evidence=[Evidence(doc.doc_id, quote)], reference_answer=str(data.get("answer", "")),
            origin="llm", reviewed=False,
        )
        out.append(q)
        if on_draft is not None:
            on_draft(q)
    return out, dict(rejected)
