"""Turn an ablation run into a results table, a quality-vs-latency chart and a failure analysis.

    python scripts/report.py runs/ablation_full [--baseline fixed__dense__norerank__minilm]

Writes results.md, results.json and quality_vs_latency.png into the run directory; they are
published as docs/results/ablation.md, ablation.json and quality_vs_latency.png. Every number comes
from the run's own summary.json / per_question.jsonl files.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

from rageval.metrics import bootstrap_ci, paired_bootstrap_diff  # noqa: E402

# Categorical slots 1-3 (blue, orange, aqua), assigned to retrievers in fixed order.
RETRIEVER_COLORS = {"dense": "#2a78d6", "bm25": "#eb6834", "hybrid": "#1baf7a"}
INK, MUTED, GRID, WHISKER = "#1f1f1e", "#6b6a64", "#e6e5df", "#c9c8c0"


def load(run_dir: Path) -> list[dict]:
    configs = []
    for summary_path in sorted(run_dir.glob("*/summary.json")):
        s = json.loads(summary_path.read_text(encoding="utf-8"))
        rows = [json.loads(x) for x in
                (summary_path.parent / "per_question.jsonl").read_text(encoding="utf-8").splitlines()]
        configs.append({"name": summary_path.parent.name, "summary": s, "rows": rows})
    return configs


def fmt(m: dict) -> str:
    return f"{m['mean']:.3f} [{m['ci_low']:.2f}, {m['ci_high']:.2f}]"


def describe(name: str) -> dict:
    """Read retriever and reranker back out of a grid config name like fixed__hybrid__ce__minilm."""
    parts = name.split("__")
    retriever = next((p for p in parts if p in RETRIEVER_COLORS), parts[0])
    return {"retriever": retriever, "rerank": "ce" if "ce" in parts else "none"}


def results_json(configs: list[dict]) -> dict:
    """Every config's headline metrics with CIs: docs/results/ablation.json, which the web view
    and the demo read."""
    first = configs[0]["summary"]
    out = {"num_questions": first["num_questions"], "num_answerable": first["num_answerable"],
           "configs": []}
    for c in sorted(configs, key=lambda c: c["name"]):
        s = c["summary"]
        chunking, retriever, rerank, embedding = c["name"].split("__")
        out["configs"].append({
            "name": c["name"], "chunking": chunking, "retriever": retriever, "rerank": rerank,
            "embedding": embedding,
            **{m: {k: s["metrics"][m][k] for k in ("mean", "ci_low", "ci_high")}
               for m in ("recall@5", "mrr@10")},
            "p50_ms": round(s["latency"]["retrieval_p50_s"] * 1000, 1),
            "num_chunks": s["num_chunks"],
        })
    return out


def chart(configs: list[dict], out: Path, metric: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    for c in configs:
        d = describe(c["name"])
        m = c["summary"]["metrics"][metric]
        x = c["summary"]["latency"]["retrieval_p50_s"] * 1000
        reranked = d.get("rerank") == "ce"
        ax.errorbar(x, m["mean"], yerr=[[m["mean"] - m["ci_low"]], [m["ci_high"] - m["mean"]]],
                    fmt="none", ecolor=WHISKER, elinewidth=1, zorder=1)
        ax.scatter(x, m["mean"], s=70, zorder=2, color=RETRIEVER_COLORS.get(d["retriever"], MUTED),
                   marker="D" if reranked else "o", edgecolor="white", linewidth=1.5)
    ax.set_xscale("log")
    xs = [c["summary"]["latency"]["retrieval_p50_s"] * 1000 for c in configs]
    ax.set_xticks([t for t in (5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000)
                   if min(xs) * 0.7 <= t <= max(xs) * 1.4])
    ax.set_xlim(min(xs) * 0.7, max(xs) * 1.4)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Retrieval latency per query, p50 (ms, log scale)", color=INK)
    ax.set_ylabel(metric, color=INK)
    ax.set_title(f"Retrieval quality vs latency ({metric}, 95% CI)", color=INK, loc="left")
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=MUTED)
    handles = [plt.Line2D([], [], marker="o", ls="", color=col, label=r, markersize=8)
               for r, col in RETRIEVER_COLORS.items()]
    handles += [plt.Line2D([], [], marker="o", ls="", color=MUTED, label="no reranker"),
                plt.Line2D([], [], marker="D", ls="", color=MUTED, label="cross-encoder")]
    ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out)


def failures(config: dict, questions: dict[str, dict]) -> list[str]:
    """Classify answerable questions the config misses entirely in its top 10."""
    lines, kinds = [], Counter()
    for r in config["rows"]:
        if not r["answerable"] or r["recall@10"] > 0:
            continue
        kind = "right page, wrong passage" if r["doc_recall@10"] > 0 else "wrong page"
        kinds[kind] += 1
        q = questions[r["qid"]]
        gold = ", ".join(e["doc_id"] for e in q["evidence"])
        top = r["retrieved"][0]["chunk_id"].split("#")[0] if r["retrieved"] else "-"
        lines.append(f"| {r['qid']} | {q['type']} | {kind} | {q['question']} | `{gold}` | `{top}` |")
    head = [f"Missed in top 10: {sum(kinds.values())} "
            f"({', '.join(f'{k}: {v}' for k, v in kinds.items()) or 'none'})", "",
            "| qid | type | failure | question | gold page | top retrieved page |",
            "|---|---|---|---|---|---|"]
    return head + lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--baseline", default=None, help="config name to compare against")
    ap.add_argument("--metric", default="recall@5")
    ap.add_argument("--questions", default="data/eval/questions.jsonl")
    args = ap.parse_args()

    configs = load(args.run_dir)
    metric = args.metric
    configs.sort(key=lambda c: c["summary"]["metrics"][metric]["mean"], reverse=True)
    best = configs[0]
    base = next((c for c in configs if c["name"] == args.baseline), configs[-1])
    questions = {q["qid"]: q for q in map(json.loads, Path(args.questions).read_text(
        encoding="utf-8").splitlines()) if q}

    n = best["summary"]["num_answerable"]
    md = [f"# Results: {args.run_dir.name}", "",
          f"{n} answerable questions ({best['summary']['num_questions']} total). "
          "Means with 95% bootstrap confidence intervals over questions.", "",
          "| config | recall@5 | mrr@10 | ndcg@10 | doc_recall@5 | p50 latency | chunks |",
          "|---|---|---|---|---|---|---|"]
    for c in configs:
        s = c["summary"]
        m = s["metrics"]
        md.append(f"| {c['name']} | {fmt(m['recall@5'])} | {fmt(m['mrr@10'])} | "
                  f"{fmt(m['ndcg@10'])} | {m['doc_recall@5']['mean']:.3f} | "
                  f"{s['latency']['retrieval_p50_s'] * 1000:.0f} ms | {s['num_chunks']} |")

    def per_q(c):
        return [r[metric] for r in c["rows"] if r["answerable"]]

    d, lo, hi = paired_bootstrap_diff(per_q(base), per_q(best))
    md += ["", f"**{best['name']}** vs **{base['name']}** on {metric}: "
               f"{d:+.3f} (paired 95% CI [{lo:+.3f}, {hi:+.3f}]).", "",
           f"## By question type ({best['name']})", "", "| type | n | recall@5 | mrr@10 |",
           "|---|---|---|---|"]
    for t, v in best["summary"]["by_type"].items():
        md.append(f"| {t} | {v['recall@5']['n']} | {fmt(v['recall@5'])} | {fmt(v['mrr@10'])} |")
    # LLM-drafted questions are written while looking at the passage, so they may share more
    # wording with it than real user questions do; compare against the hand-written subset.
    md += ["", f"## By question origin ({metric})", "", "| config | hand-written | LLM-drafted |",
           "|---|---|---|"]
    for c in (best, base):
        by = {o: [r[metric] for r in c["rows"] if r["answerable"]
                  and questions[r["qid"]]["origin"] == o] for o in ("hand", "llm")}
        cells = [f"{fmt(dict(zip(('mean', 'ci_low', 'ci_high'), bootstrap_ci(v))))} (n={len(v)})"
                 for v in by.values()]
        md.append(f"| {c['name']} | {cells[0]} | {cells[1]} |")
    md += ["", f"## Failure analysis ({best['name']})", "", *failures(best, questions), "",
           "![quality vs latency](quality_vs_latency.png)", ""]
    (args.run_dir / "results.md").write_text("\n".join(md), encoding="utf-8")
    (args.run_dir / "results.json").write_text(json.dumps(results_json(configs), indent=1) + "\n",
                                               encoding="utf-8", newline="\n")
    chart(configs, args.run_dir / "quality_vs_latency.png", metric)
    print(f"wrote {args.run_dir / 'results.md'}, results.json and quality_vs_latency.png")


if __name__ == "__main__":
    main()
