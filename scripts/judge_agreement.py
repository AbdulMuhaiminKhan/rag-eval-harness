"""Compare several judges that graded the same stored answers (see `rageval rejudge`).

    python scripts/judge_agreement.py qwen-v1=runs/answers_best qwen-v2=runs/answers_best_v2 \
        gemma-v1=runs/answers_best_gemma > docs/results/judges.md

Only answerable questions the system did not abstain on are compared, since the others are
graded by rule, not by the judge.
"""

import itertools
import json
import sys
from pathlib import Path

from rageval.judge import cohens_kappa
from rageval.metrics import bootstrap_ci

LABELS = ("correct", "faithful")


def load(path: str) -> dict:
    lines = (Path(path) / "per_question.jsonl").read_text(encoding="utf-8").splitlines()
    return {r["qid"]: r for r in map(json.loads, lines)}


def main() -> None:
    runs = dict(arg.split("=", 1) for arg in sys.argv[1:])
    rows = {name: load(path) for name, path in runs.items()}
    first = next(iter(rows.values()))
    qids = [q for q, r in first.items() if r["answerable"] and not r["judge_abstained"]]

    print(f"Judged answers: {len(qids)} answerable questions where the system gave an answer.\n")
    print("| judge | correct rate [95% CI] | faithful rate [95% CI] |")
    print("|---|---|---|")
    for name, r in rows.items():
        cells = []
        for label in LABELS:
            m, lo, hi = bootstrap_ci([r[q][f"judge_{label}"] for q in qids])
            cells.append(f"{m:.3f} [{lo:.2f}, {hi:.2f}]")
        print(f"| {name} | " + " | ".join(cells) + " |")

    print("\n| judges | label | agreement | Cohen's kappa |")
    print("|---|---|---|---|")
    for (a, ra), (b, rb) in itertools.combinations(rows.items(), 2):
        for label in LABELS:
            x = [ra[q][f"judge_{label}"] for q in qids]
            y = [rb[q][f"judge_{label}"] for q in qids]
            agree = sum(i == j for i, j in zip(x, y)) / len(qids)
            print(f"| {a} vs {b} | {label} | {agree:.3f} | {cohens_kappa(x, y):.3f} |")


if __name__ == "__main__":
    main()
