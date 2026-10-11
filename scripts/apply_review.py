"""Apply a batch of review decisions to drafted questions (the non-interactive `rageval review`).

    python scripts/apply_review.py decisions.json --reviewer claude

decisions.json maps qid -> {"accept": bool, "reason": str, optional "type", "question",
"reference_answer"}. Accepted drafts are appended to data/eval/questions.jsonl; every decision
(accepted or not) is appended to data/eval/review_log.jsonl so the eval set's provenance is
auditable. Drafts whose quote repeats across several pages are rejected automatically.
"""

import argparse
import json
from pathlib import Path

from rageval.ingest import load_corpus
from rageval.qgen import pages_containing

CANDIDATES = Path("data/eval/candidates.jsonl")
QUESTIONS = Path("data/eval/questions.jsonl")
LOG = Path("data/eval/review_log.jsonl")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("decisions", type=Path)
    ap.add_argument("--reviewer", required=True)
    args = ap.parse_args()

    decisions = json.loads(args.decisions.read_text(encoding="utf-8"))
    docs = load_corpus(Path("data/processed/corpus.jsonl"))
    existing = {json.loads(x)["qid"] for x in QUESTIONS.read_text(encoding="utf-8").splitlines() if x}
    remaining, accepted, log = [], [], []
    for line in CANDIDATES.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        q = json.loads(line)
        d = decisions.get(q["qid"])
        if d is None or q["qid"] in existing:
            remaining.append(line)
            continue
        accept, reason = d["accept"], d.get("reason", "")
        if accept and pages_containing(docs, q["evidence"][0]["quote"]) > 1:
            accept, reason = False, "quote repeats on several pages (boilerplate)"
        for key in ("type", "question", "reference_answer"):
            if key in d:
                q[key] = d[key]
        log.append({"qid": q["qid"], "accepted": accept, "reason": reason,
                    "reviewer": args.reviewer, "question": q["question"]})
        if accept:
            q["reviewed"], q["reviewer"] = True, args.reviewer
            accepted.append(q)

    with QUESTIONS.open("a", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(q, ensure_ascii=False) + "\n" for q in accepted)
    with LOG.open("a", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in log)
    CANDIDATES.write_text("".join(x + "\n" for x in remaining), encoding="utf-8", newline="\n")
    print(f"accepted {len(accepted)}, rejected {len(log) - len(accepted)}, "
          f"{len(remaining)} drafts left to review")


if __name__ == "__main__":
    main()
