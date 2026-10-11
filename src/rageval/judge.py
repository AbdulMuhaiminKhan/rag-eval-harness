"""LLM-as-judge for answers, and agreement statistics for calibrating it against humans."""

from __future__ import annotations

import json
from collections.abc import Sequence

from .generation import LLM, format_context, is_abstention
from .retrieval import Hit

JUDGE_PROMPT = """You are grading an answer produced by a question-answering system over Kubernetes docs.

Question: {question}
Reference answer: {reference}

Context the system was given:
{context}

System answer: {answer}

Grade two things:
- "faithful": 1 if every factual claim in the system answer is supported by the context, else 0.
- "correct": 1 if the system answer agrees with the reference answer on the key facts
  (extra correct detail is fine), else 0.
Respond with JSON only: {{"faithful": 0 or 1, "correct": 0 or 1, "reason": "<one sentence>"}}"""


def judge_answer(llm: LLM, question: str, reference: str, hits: list[Hit], answer: str,
                 answerable: bool) -> dict:
    abstained = is_abstention(answer)
    if not answerable:
        # Unanswerable: the only correct behaviour is to abstain; no judge call needed.
        return {"faithful": int(abstained), "correct": int(abstained), "abstained": abstained,
                "reason": "rule: unanswerable question"}
    if abstained:
        return {"faithful": 1, "correct": 0, "abstained": True, "reason": "rule: abstained"}
    res = llm.complete(JUDGE_PROMPT.format(question=question, reference=reference,
                                           context=format_context(hits), answer=answer),
                       json_mode=True)
    try:
        data = json.loads(res.text)
        out = {"faithful": int(data["faithful"]), "correct": int(data["correct"]),
               "reason": str(data.get("reason", ""))}
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        out = {"faithful": None, "correct": None, "reason": f"unparseable judge output: {res.text[:200]}"}
    out["abstained"] = False
    return out


def cohens_kappa(a: Sequence[int], b: Sequence[int]) -> float:
    """Agreement beyond chance between two binary labelers."""
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def agreement(human: Sequence[int], judge: Sequence[int]) -> dict:
    n = len(human)
    return {
        "n": n,
        "accuracy": sum(h == j for h, j in zip(human, judge)) / n if n else float("nan"),
        "kappa": cohens_kappa(human, judge),
        "judge_positive_rate": sum(judge) / n if n else float("nan"),
        "human_positive_rate": sum(human) / n if n else float("nan"),
    }
