"""Draft eval questions with an LLM, for a human to review before they enter the eval set.

The LLM sees one passage and must return a question plus a verbatim quote from the passage
that answers it. Drafts whose quote cannot be found in the source document are dropped.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict

from .chunking import fixed_chunks
from .dataset import QUESTION_TYPES, Evidence, Question, find_span
from .generation import LLM
from .ingest import Document

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


def draft_questions(llm: LLM, docs: dict[str, Document], n: int, seed: int = 0,
                    start_index: int = 1) -> list[Question]:
    rng = random.Random(seed)
    # Stratify by top-level section (concepts, tasks, reference, ...) so drafts cover the corpus.
    by_section: dict[str, list[Document]] = defaultdict(list)
    for d in docs.values():
        by_section[d.doc_id.split("/")[0]].append(d)
    sections = sorted(by_section)
    target_types = [t for t in QUESTION_TYPES if t not in ("unanswerable", "multi_hop")]

    out: list[Question] = []
    attempts = 0
    while len(out) < n and attempts < n * 3:
        attempts += 1
        doc = rng.choice(by_section[sections[attempts % len(sections)]])
        chunks = [c for c in fixed_chunks(doc, size=1200, overlap=0) if len(c.text) > 400]
        if not chunks:
            continue
        passage = rng.choice(chunks).text
        qtype = target_types[attempts % len(target_types)]
        res = llm.complete(QGEN_PROMPT.format(doc_id=doc.doc_id, qtype=qtype, passage=passage),
                           json_mode=True)
        try:
            data = json.loads(res.text)
            question, quote = data["question"].strip(), data["quote"].strip()
        except (json.JSONDecodeError, KeyError, AttributeError):
            continue
        if not question or not quote or find_span(doc.text, quote) is None:
            continue
        typ = data.get("type") if data.get("type") in QUESTION_TYPES else qtype
        out.append(Question(
            qid=f"g{start_index + len(out):04d}", question=question, type=typ, answerable=True,
            evidence=[Evidence(doc.doc_id, quote)], reference_answer=str(data.get("answer", "")),
            origin="llm", reviewed=False,
        ))
    return out
