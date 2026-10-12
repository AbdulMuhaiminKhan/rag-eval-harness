"""Command line entry point: ``rageval --help``."""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import typer

from .config import Config, GenerationConfig, expand_grid
from .dataset import Question, load_questions
from .ingest import build_corpus, load_corpus

app = typer.Typer(add_completion=False, no_args_is_help=True)

DEFAULT_CONFIG = "configs/baseline.yaml"
DEFAULT_QUESTIONS = "data/eval/questions.jsonl"


def _run_dir(name: str, out: str | None) -> Path:
    return Path(out) if out else Path("runs") / f"{time.strftime('%Y%m%d-%H%M%S')}_{name}"


def _print_summary(s: dict) -> None:
    for m in ("recall@1", "recall@5", "recall@10", "mrr@10", "ndcg@10", "doc_recall@5",
              "correct", "faithful", "abstention_on_unanswerable"):
        v = s["metrics"].get(m)
        if v:
            typer.echo(f"  {m:<28} {v['mean']:.3f}  [{v['ci_low']:.3f}, {v['ci_high']:.3f}]"
                       f"  n={v['n']}")
    typer.echo(f"  retrieval p50 latency       {s['latency']['retrieval_p50_s'] * 1000:.0f} ms")


@app.command()
def ingest(raw_dir: str = "data/raw", out_dir: str = "data/processed", ref: str = "main"):
    """Download the Kubernetes docs and write data/processed/corpus.jsonl."""
    m = build_corpus(Path(raw_dir), Path(out_dir), ref)
    typer.echo(json.dumps(m, indent=2))


@app.command()
def sample_corpus(questions: str = DEFAULT_QUESTIONS, corpus: str = "data/processed/corpus.jsonl",
                  out: str = "data/processed/corpus_ci.jsonl", distractors: int = 300,
                  seed: int = 0):
    """Write a small corpus (gold docs + seeded random distractors) for fast CI runs."""
    docs = load_corpus(Path(corpus))
    qs, _ = load_questions(Path(questions), docs, strict=False)
    gold = {s.doc_id for q in qs for e in q.evidence for s in e.spans()}
    rest = sorted(set(docs) - gold)
    keep = gold | set(random.Random(seed).sample(rest, min(distractors, len(rest))))
    with open(corpus, encoding="utf-8") as fin, open(out, "w", encoding="utf-8", newline="\n") as fout:
        for line in fin:
            if json.loads(line)["doc_id"] in keep:
                fout.write(line)
    typer.echo(f"wrote {len(keep)} docs ({len(gold)} gold) to {out}")


@app.command()
def index(config: str = DEFAULT_CONFIG):
    """Chunk the corpus and build the indexes a config needs."""
    from .retrieval import Retriever

    r = Retriever(Config.load(config))
    typer.echo(f"{len(r.chunks)} chunks indexed")


@app.command()
def ask(question: str, config: str = DEFAULT_CONFIG, show_context: bool = False):
    """Answer one question with the configured pipeline."""
    from .generation import LLM, answer
    from .retrieval import Retriever

    cfg = Config.load(config)
    hits = Retriever(cfg).retrieve(question)[: cfg.generation.context_k]
    for i, h in enumerate(hits, 1):
        typer.echo(f"[{i}] {h.chunk.doc_id}  score={h.score:.3f}")
        if show_context:
            typer.echo(h.chunk.text[:500] + "\n")
    res = answer(LLM(cfg.generation), question, hits)
    typer.echo(f"\n{res.text}\n\n({res.latency_s:.1f}s, {res.tokens_in}+{res.tokens_out} tokens)")


@app.command()
def validate(questions: str = DEFAULT_QUESTIONS, corpus: str = "data/processed/corpus.jsonl"):
    """Check every evidence quote resolves to a span in the corpus."""
    qs, problems = load_questions(Path(questions), load_corpus(Path(corpus)), strict=False)
    for p in problems:
        typer.echo(f"PROBLEM {p}")
    counts: dict[str, int] = {}
    for q in qs:
        counts[q.type] = counts.get(q.type, 0) + 1
    typer.echo(f"{len(qs)} questions, {sum(q.reviewed for q in qs)} reviewed, by type: {counts}")
    raise typer.Exit(1 if problems else 0)


@app.command("eval")
def eval_cmd(config: str = DEFAULT_CONFIG, questions: str = DEFAULT_QUESTIONS,
             corpus: str | None = None, generate: bool = False, reviewed_only: bool = False,
             limit: int | None = None, out: str | None = None,
             resume: bool = typer.Option(False, help="continue an interrupted run in --out")):
    """Evaluate one config. Retrieval metrics always; answers + judge with --generate."""
    from .evaluate import run_eval

    cfg = Config.load(config)
    if corpus:
        cfg.corpus = corpus
    qs, problems = load_questions(Path(questions), load_corpus(Path(cfg.corpus)), reviewed_only)
    if problems:
        typer.echo(f"skipping {len(problems)} questions with unresolved evidence (see validate)")
    qs = qs[:limit] if limit else qs
    out_dir = _run_dir(cfg.name, out)
    s = run_eval(cfg, qs, out_dir, generate=generate, resume=resume)
    typer.echo(f"{cfg.name}: {s['num_chunks']} chunks -> {out_dir}")
    _print_summary(s)


@app.command()
def ablate(grid: str = "configs/ablation.yaml", questions: str = DEFAULT_QUESTIONS,
           generate: bool = False, reviewed_only: bool = False, out: str | None = None):
    """Run every config in an ablation grid and write a combined table."""
    from .evaluate import run_eval

    out_root = _run_dir("ablation", out)
    rows = []
    # Load the eval set once per corpus, so every config is scored on the same questions even
    # if the file is edited while a long grid is running.
    question_sets: dict[str, list[Question]] = {}
    for cfg in expand_grid(grid):
        if cfg.corpus not in question_sets:
            question_sets[cfg.corpus], _ = load_questions(
                Path(questions), load_corpus(Path(cfg.corpus)), reviewed_only)
        s = run_eval(cfg, question_sets[cfg.corpus], out_root / cfg.name, generate=generate)
        rows.append({"name": cfg.name, **{k: v["mean"] for k, v in s["metrics"].items()},
                     **s["latency"], "num_chunks": s["num_chunks"]})
        typer.echo(cfg.name)
        _print_summary(s)
    (out_root / "ablation.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    typer.echo(f"wrote {out_root / 'ablation.json'}")


@app.command()
def gen_questions(n: int = 50, out: str = "data/eval/candidates.jsonl", seed: int = 0,
                  model: str = "qwen2.5:7b", corpus: str = "data/processed/corpus.jsonl"):
    """Draft questions with a local LLM into a candidates file for human review."""
    from .generation import LLM
    from .qgen import draft_questions

    records = [json.loads(x) for path in (out, DEFAULT_QUESTIONS, "data/eval/review_log.jsonl")
               if Path(path).exists()
               for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    prior = [r["question"] for r in records]
    # Draft ids must never be reused, including ids of drafts that were already rejected.
    next_id = max((int(r["qid"][1:]) for r in records if r["qid"].startswith("g")), default=0) + 1
    llm = LLM(GenerationConfig(model=model, temperature=0.7))
    with open(out, "a", encoding="utf-8", newline="\n") as f:

        def save(q: Question) -> None:
            f.write(json.dumps(q.to_dict(), ensure_ascii=False) + "\n")
            f.flush()
            typer.echo(f"{q.qid} [{q.type}] {q.question}")

        drafts, rejected = draft_questions(llm, load_corpus(Path(corpus)), n, seed,
                                           next_id, existing=prior, on_draft=save)
    typer.echo(f"appended {len(drafts)} drafts to {out}; rejected: {rejected}")


@app.command()
def review(candidates: str = "data/eval/candidates.jsonl", questions: str = DEFAULT_QUESTIONS,
           reviewer: str = typer.Option(..., help="your name, recorded on accepted questions")):
    """Interactively accept, edit or reject drafted questions."""
    lines = Path(candidates).read_text(encoding="utf-8").splitlines()
    done = set()
    if Path(questions).exists():
        done = {json.loads(x)["qid"] for x in Path(questions).read_text(encoding="utf-8").splitlines()
                if x.strip()}
    remaining = []
    for i, line in enumerate(lines):
        q = Question.from_dict(json.loads(line))
        if q.qid in done:
            continue
        ev = q.evidence[0]
        typer.echo(f"\n--- {q.qid} ({i + 1}/{len(lines)}) [{q.type}] {ev.doc_id}")
        typer.echo(f"Q: {q.question}\nA: {q.reference_answer}\nEvidence: {ev.quote}")
        choice = typer.prompt("[a]ccept [e]dit [r]eject [s]kip [q]uit", default="s")
        if choice == "q":
            remaining.extend(lines[i:])
            break
        if choice in ("a", "e"):
            if choice == "e":
                q.question = typer.prompt("question", default=q.question)
                q.reference_answer = typer.prompt("answer", default=q.reference_answer)
                q.type = typer.prompt("type", default=q.type)
            q.reviewed, q.reviewer = True, reviewer
            with open(questions, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(q.to_dict(), ensure_ascii=False) + "\n")
        elif choice == "s":
            remaining.append(line)
    Path(candidates).write_text("\n".join(remaining) + ("\n" if remaining else ""),
                                encoding="utf-8")


@app.command()
def gate(run: str = "runs/ci", baseline: str = "ci/baseline.json", max_drop: float = 0.03):
    """Fail (exit 1) if any gated metric dropped more than max_drop below the baseline."""
    s = json.loads((Path(run) / "summary.json").read_text(encoding="utf-8"))
    b = json.loads(Path(baseline).read_text(encoding="utf-8"))
    stale = [k for k, v in b.get("fingerprint", {}).items() if s.get("fingerprint", {}).get(k) != v]
    if stale or "fingerprint" not in b:
        typer.echo(f"STALE baseline: {', '.join(stale) or 'no fingerprint'} changed since "
                   f"{baseline} was made, so the numbers are not comparable. If the change is "
                   "intended, regenerate it with scripts/update_ci_baseline.py in the same PR.")
        raise typer.Exit(2)
    failed = False
    for m, base in b["metrics"].items():
        cur = s["metrics"][m]["mean"]
        ok = cur >= base - max_drop
        failed |= not ok
        typer.echo(f"{'OK  ' if ok else 'FAIL'} {m:<12} {cur:.3f} (baseline {base:.3f}, "
                   f"allowed drop {max_drop:.3f})")
    raise typer.Exit(1 if failed else 0)


@app.command()
def rejudge(run: str = typer.Option(..., help="a --generate run dir"),
            out: str = typer.Option(..., help="where to write the re-judged run"),
            version: str = "v1",
            model: str | None = typer.Option(None, help="judge model; default: the generator")):
    """Re-grade the stored answers of a --generate run with another judge prompt version,
    without regenerating them, so judge variants are compared on identical answers."""
    from tqdm import tqdm

    from .evaluate import summarize
    from .generation import LLM
    from .judge import judge_answer
    from .retrieval import Hit, load_or_build_chunks

    src = Path(run)
    summary = json.loads((src / "summary.json").read_text(encoding="utf-8"))
    cfg = Config.from_dict(summary["config"])
    chunks = {c.chunk_id: c for c in load_or_build_chunks(cfg)}
    qs, _ = load_questions(Path(DEFAULT_QUESTIONS), load_corpus(Path(cfg.corpus)), strict=False)
    by_id = {q.qid: q for q in qs}
    judge_cfg = GenerationConfig(**{**asdict(cfg.generation), **({"model": model} if model else {})})
    llm = LLM(judge_cfg)
    rows = [json.loads(x) for x in (src / "per_question.jsonl").read_text(encoding="utf-8").splitlines()]
    dst = Path(out)
    dst.mkdir(parents=True, exist_ok=True)
    with (dst / "per_question.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for r in tqdm(rows, desc=f"judge {version}"):
            q = by_id[r["qid"]]
            hits = [Hit(chunks[h["chunk_id"]], h["score"])
                    for h in r["retrieved"][: cfg.generation.context_k]]
            verdict = judge_answer(llm, q.question, q.reference_answer, hits, r["answer"],
                                   q.answerable, version=version)
            r.update({f"judge_{k}": v for k, v in verdict.items()})
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
    new = summarize(rows)
    for key in ("config", "index_build_s", "num_chunks", "fingerprint"):
        new[key] = summary.get(key)
    new["judge"] = {"prompt": version, "model": judge_cfg.model}
    (dst / "summary.json").write_text(json.dumps(new, indent=2), encoding="utf-8")
    _print_summary(new)


@app.command()
def calibrate(run: list[str] = typer.Option(..., help="one or more judged runs to compare"),
              labels: str = "data/eval/human_labels.jsonl"):
    """Report each judge's agreement with human labels (accuracy and Cohen's kappa)."""
    from .judge import agreement

    human = [json.loads(x) for x in Path(labels).read_text(encoding="utf-8").splitlines()
             if x.strip()]
    typer.echo(f"{'run':<34} {'label':<9} {'n':>3} {'acc':>6} {'kappa':>6} {'judge+':>7} {'human+':>7}")
    for r in run:
        judged = {}
        for line in (Path(r) / "per_question.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            judged[row["qid"]] = row
        for key in ("correct", "faithful"):
            pairs = [(h[key], judged[h["qid"]][f"judge_{key}"]) for h in human
                     if h.get(key) is not None and h["qid"] in judged
                     and judged[h["qid"]].get(f"judge_{key}") is not None]
            res = agreement([p[0] for p in pairs], [p[1] for p in pairs])
            typer.echo(f"{Path(r).name:<34} {key:<9} {res['n']:>3} {res['accuracy']:>6.3f} "
                       f"{res['kappa']:>6.3f} {res['judge_positive_rate']:>7.3f} "
                       f"{res['human_positive_rate']:>7.3f}")


if __name__ == "__main__":
    app()
