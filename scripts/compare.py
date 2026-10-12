"""Paired comparisons that isolate one design choice at a time in an ablation run.

    python scripts/compare.py runs/ablation_full > docs/results/comparisons.md

Each row changes exactly one axis and reports the mean per-question difference with a paired
bootstrap 95% CI. An interval that crosses 0 means the eval set cannot tell the two apart.
"""

import json
import sys
from pathlib import Path

from rageval.metrics import paired_bootstrap_diff

PAIRS = [
    ("cross-encoder reranker, on dense", "fixed__dense__norerank__minilm", "fixed__dense__ce__minilm"),
    ("cross-encoder reranker, on hybrid", "fixed__hybrid__norerank__minilm",
     "fixed__hybrid__ce__minilm"),
    ("hybrid instead of dense", "fixed__dense__norerank__minilm", "fixed__hybrid__norerank__minilm"),
    ("hybrid instead of BM25", "fixed__bm25__norerank__minilm", "fixed__hybrid__norerank__minilm"),
    ("bge-small instead of MiniLM, dense", "fixed__dense__norerank__minilm",
     "fixed__dense__norerank__bge-small"),
    ("semantic instead of fixed chunks, dense", "fixed__dense__norerank__minilm",
     "semantic__dense__norerank__minilm"),
    ("semantic instead of fixed chunks, hybrid bge", "fixed__hybrid__norerank__bge-small",
     "semantic__hybrid__norerank__bge-small"),
    ("semantic instead of fixed chunks, best config", "fixed__hybrid__ce__minilm",
     "semantic__hybrid__ce__minilm"),
]
METRICS = ("recall@5", "mrr@10")


def load(run: Path, name: str) -> dict:
    rows = map(json.loads, (run / name / "per_question.jsonl").read_text(encoding="utf-8").split("\n")[:-1])
    return {r["qid"]: r for r in rows if r["answerable"]}


def main() -> None:
    """Print a Markdown table; with --json PATH also write the same numbers as JSON."""
    run = Path(sys.argv[1])
    json_out = Path(sys.argv[sys.argv.index("--json") + 1]) if "--json" in sys.argv else None
    print("| change | " + " | ".join(f"Δ {m} [95% CI]" for m in METRICS) + " |")
    print("|---|" + "---|" * len(METRICS))
    records = []
    for label, a, b in PAIRS:
        base, new = load(run, a), load(run, b)
        qids = sorted(base)
        cells, rec = [], {"change": label, "from": a, "to": b}
        for m in METRICS:
            d, lo, hi = paired_bootstrap_diff([base[q][m] for q in qids], [new[q][m] for q in qids])
            cells.append(f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
            rec[m] = {"diff": round(d, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4)}
        records.append(rec)
        print(f"| {label} | " + " | ".join(cells) + " |")
    if json_out:
        json_out.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
