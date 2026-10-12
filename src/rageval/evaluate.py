"""Run a config over an eval set and write per-question results plus a summary with CIs."""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .config import Config
from .dataset import Question
from .generation import LLM, answer
from .judge import judge_answer
from .metrics import bootstrap_ci, retrieval_scores
from .retrieval import Retriever

KS = (1, 3, 5, 10)
HEADLINE = ("recall@5", "mrr@10", "ndcg@10")


def run_eval(cfg: Config, questions: list[Question], out_dir: Path, generate: bool = False,
             judge_cfg=None, min_overlap: float = 0.5, resume: bool = False) -> dict:
    """Evaluate ``questions``. Rows are appended to per_question.jsonl as they finish, so a long
    --generate run that is interrupted can continue with ``resume=True``."""
    cfg.retrieval.top_k = max(cfg.retrieval.top_k, max(KS))
    t0 = time.perf_counter()
    retriever = Retriever(cfg)
    build_s = time.perf_counter() - t0
    # Warm up (model load, CUDA init, first HNSW query) so per-query latency is steady state.
    retriever.retrieve("warm-up query")
    llm = judge_llm = None
    if generate:
        llm = LLM(cfg.generation)
        judge_llm = LLM(judge_cfg or cfg.generation)

    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "per_question.jsonl"
    rows = []
    if resume and rows_path.exists():
        wanted = {q.qid for q in questions}
        rows = [r for r in map(json.loads, rows_path.read_text(encoding="utf-8").splitlines())
                if r["qid"] in wanted]
    done = {r["qid"] for r in rows}
    out = rows_path.open("w", encoding="utf-8", newline="\n")
    out.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    for q in tqdm([q for q in questions if q.qid not in done], desc=cfg.name):
        t = time.perf_counter()
        hits = retriever.retrieve(q.question)
        row = {"qid": q.qid, "type": q.type, "answerable": q.answerable,
               "retrieval_latency_s": time.perf_counter() - t,
               "retrieved": [{"chunk_id": h.chunk.chunk_id, "score": h.score} for h in hits]}
        if q.answerable:
            row.update(retrieval_scores([h.chunk for h in hits], q.evidence, KS, min_overlap))
        if llm is not None:
            context = hits[: cfg.generation.context_k]
            res = answer(llm, q.question, context)
            row.update({"answer": res.text, "generation_latency_s": res.latency_s,
                        "tokens_in": res.tokens_in, "tokens_out": res.tokens_out,
                        "cost_usd": res.cost_usd})
            row.update({f"judge_{k}": v for k, v in judge_answer(
                judge_llm, q.question, q.reference_answer, context, res.text,
                q.answerable).items()})
        rows.append(row)
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        out.flush()
    out.close()

    summary = summarize(rows)
    summary["config"] = asdict(cfg)
    summary["index_build_s"] = build_s
    summary["num_chunks"] = len(retriever.chunks)
    summary["fingerprint"] = fingerprint(cfg, questions)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def fingerprint(cfg: Config, questions: list[Question]) -> dict:
    """What a run's numbers depend on, independent of OS line endings and file order. Two runs
    are only comparable (e.g. by the CI gate) when their fingerprints match."""
    corpus_lines = sorted(Path(cfg.corpus).read_text(encoding="utf-8").splitlines())
    pipeline = {"chunking": asdict(cfg.chunking), "retrieval": asdict(cfg.retrieval),
                "rerank": asdict(cfg.rerank)}
    return {
        "questions": _sha(json.dumps(sorted((q.to_dict() for q in questions),
                                            key=lambda d: d["qid"]), sort_keys=True)),
        "corpus": _sha("\n".join(corpus_lines)),
        "pipeline": _sha(json.dumps(pipeline, sort_keys=True)),
    }


def _ci(values: list[float]) -> dict:
    m, lo, hi = bootstrap_ci(values)
    return {"mean": m, "ci_low": lo, "ci_high": hi, "n": len(values)}


def summarize(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["answerable"]]
    metric_names = [k for k in (answerable[0] if answerable else {}) if "@" in k]
    out: dict = {"num_questions": len(rows), "num_answerable": len(answerable), "metrics": {},
                 "by_type": {}}
    for m in metric_names:
        out["metrics"][m] = _ci([r[m] for r in answerable])
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in answerable:
        by_type[r["type"]].append(r)
    for t, rs in sorted(by_type.items()):
        out["by_type"][t] = {m: _ci([r[m] for r in rs]) for m in HEADLINE if m in rs[0]}
    lat = [r["retrieval_latency_s"] for r in rows]
    out["latency"] = {"retrieval_p50_s": float(np.median(lat)),
                      "retrieval_p95_s": float(np.percentile(lat, 95))}
    if rows and "answer" in rows[0]:
        gen = [r["generation_latency_s"] for r in rows]
        out["latency"].update({"generation_p50_s": float(np.median(gen)),
                               "generation_p95_s": float(np.percentile(gen, 95))})
        out["cost_per_query_usd"] = float(np.mean([r["cost_usd"] for r in rows]))
        for key in ("judge_correct", "judge_faithful"):
            vals = [r[key] for r in rows if r.get(key) is not None]
            out["metrics"][key.removeprefix("judge_")] = _ci(vals)
        unans = [r for r in rows if not r["answerable"]]
        if unans:
            out["metrics"]["abstention_on_unanswerable"] = _ci(
                [int(r["judge_abstained"]) for r in unans])
            out["metrics"]["false_abstention_on_answerable"] = _ci(
                [int(r["judge_abstained"]) for r in answerable])
    return out
