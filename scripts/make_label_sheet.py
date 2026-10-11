"""Sample judged answers from a --generate run into a sheet for a human to label.

    python scripts/make_label_sheet.py runs/answers_best --n 40

Writes data/eval/human_labels.jsonl with "correct" and "faithful" set to null. Fill each with
0 or 1 by reading the question, reference answer, retrieved context and system answer (the
judge's verdict is deliberately left out so it cannot anchor you), then run:

    rageval calibrate --run runs/answers_best
"""

import argparse
import json
import random
from pathlib import Path

from rageval.config import Config
from rageval.retrieval import load_or_build_chunks

OUT = Path("data/eval/human_labels.jsonl")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(x) for x in (args.run / "per_question.jsonl").read_text(
        encoding="utf-8").splitlines() if x]
    judged = [r for r in rows if r.get("judge_correct") is not None
              and not r["judge_reason"].startswith("rule:")]
    sample = sorted(random.Random(args.seed).sample(judged, min(args.n, len(judged))),
                    key=lambda r: r["qid"])

    cfg = Config.from_dict(json.loads((args.run / "summary.json").read_text(
        encoding="utf-8"))["config"])
    chunks = {c.chunk_id: c for c in load_or_build_chunks(cfg)}
    questions = {q["qid"]: q for q in map(json.loads, Path("data/eval/questions.jsonl").read_text(
        encoding="utf-8").splitlines()) if q}

    with OUT.open("w", encoding="utf-8", newline="\n") as f:
        for r in sample:
            q = questions[r["qid"]]
            context = [chunks[h["chunk_id"]].text for h in r["retrieved"][: cfg.generation.context_k]]
            f.write(json.dumps({
                "qid": r["qid"], "correct": None, "faithful": None,
                "question": q["question"], "reference_answer": q["reference_answer"],
                "answer": r["answer"], "context": context,
            }, ensure_ascii=False) + "\n")
    print(f"wrote {len(sample)} items to {OUT}; fill in correct/faithful with 0 or 1")


if __name__ == "__main__":
    main()
