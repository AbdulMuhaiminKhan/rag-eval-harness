"""Regenerate ci/baseline.json from a CI-config eval run.

    rageval sample-corpus --distractors 300
    rageval eval --config configs/ci.yaml --out runs/ci
    python scripts/update_ci_baseline.py runs/ci
"""

import json
import sys
from pathlib import Path

GATED = ("recall@5", "mrr@10", "ndcg@10")

run = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/ci")
summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
baseline = {
    "num_questions": summary["num_questions"],
    "num_chunks": summary["num_chunks"],
    "metrics": {m: round(summary["metrics"][m]["mean"], 4) for m in GATED},
}
Path("ci").mkdir(exist_ok=True)
Path("ci/baseline.json").write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
print(json.dumps(baseline, indent=2))
