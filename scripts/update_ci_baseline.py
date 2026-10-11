"""Regenerate ci/baseline.json exactly the way CI evaluates (CPU only, CI corpus slice).

    python scripts/update_ci_baseline.py

Run this in the same PR whenever the eval set, the corpus or the CI pipeline config changes on
purpose; the gate refuses to compare runs whose fingerprints differ from the baseline.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

GATED = ("recall@5", "mrr@10", "ndcg@10")
RUN = Path("runs/ci")

env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}  # CI runners have no GPU; match them
rageval = [sys.executable, "-m", "rageval.cli"]
subprocess.run([*rageval, "sample-corpus", "--distractors", "300"], check=True, env=env)
subprocess.run([*rageval, "eval", "--config", "configs/ci.yaml", "--out", str(RUN)],
               check=True, env=env)

summary = json.loads((RUN / "summary.json").read_text(encoding="utf-8"))
baseline = {
    "num_questions": summary["num_questions"],
    "num_chunks": summary["num_chunks"],
    "fingerprint": summary["fingerprint"],
    "metrics": {m: round(summary["metrics"][m]["mean"], 4) for m in GATED},
}
Path("ci").mkdir(exist_ok=True)
Path("ci/baseline.json").write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8",
                                    newline="\n")
print(json.dumps(baseline, indent=2))
