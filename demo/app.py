"""Web demo: ask the Kubernetes docs, see what each retrieval setup finds and how long it took.

    python demo/app.py            # local, http://127.0.0.1:7860
    RAGEVAL_DEMO_GENERATE=1 python demo/app.py   # also generate an answer through Ollama

Runs the best config from the ablation (hybrid BM25 + MiniLM, cross-encoder rerank) next to the
dense-only baseline, so the effect of the design choices is visible on any question. Generation
is off by default because a free CPU host cannot run the 7B model.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from rageval.config import Config  # noqa: E402
from rageval.ingest import build_corpus, load_corpus  # noqa: E402
from rageval.retrieval import Retriever  # noqa: E402
from rageval.tracing import Trace  # noqa: E402

CORPUS = Path(os.environ.get("RAGEVAL_DEMO_CORPUS", "data/processed/corpus.jsonl"))
GENERATE = os.environ.get("RAGEVAL_DEMO_GENERATE") == "1"
EXAMPLES = [
    "How do I roll back a Deployment to the previous version?",
    "What happens to emptyDir data when a node is drained?",
    "Why does my CronJob skip runs?",
    "How can I keep a pod off nodes that are under maintenance?",
    "What is the difference between a Role and a ClusterRole?",
]


def ensure_corpus() -> None:
    if not CORPUS.exists():
        commit = (ROOT / "data" / "CORPUS_COMMIT").read_text(encoding="utf-8").strip()
        build_corpus(Path("data/raw"), CORPUS.parent, commit)


def load(config: str) -> tuple[Config, Retriever]:
    cfg = Config.load(config)
    cfg.corpus = str(CORPUS)
    cfg.retrieval.top_k = 5
    return cfg, Retriever(cfg)


def header() -> str:
    """Corpus size and eval numbers, read from the corpus and docs/results/ablation.json so the
    page never disagrees with the committed results."""
    manifest = CORPUS.parent / "manifest.json"
    meta = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
    pages = meta.get("num_docs") or len(load_corpus(CORPUS))
    version = f" ({meta['kubernetes_version']})" if meta.get("kubernetes_version") else ""
    ablation = json.loads((ROOT / "docs" / "results" / "ablation.json").read_text(encoding="utf-8"))
    cfgs = ablation["configs"]
    best = max(cfgs, key=lambda c: c["recall@5"]["mean"])["recall@5"]["mean"]
    base = next(c for c in cfgs if c["name"] == "fixed__dense__norerank__minilm")["recall@5"]["mean"]
    return (
        "# Ask the Kubernetes docs\n"
        f"Retrieval over {pages:,} pages of the Kubernetes documentation{version}. The left column "
        f"is the best setup from a {len(cfgs)}-config ablation (hybrid BM25 + embeddings, "
        f"cross-encoder rerank, Recall@5 {best:.2f}); the right is the dense-only baseline "
        f"(Recall@5 {base:.2f}). "
        "[Code and results](https://github.com/AbdulMuhaiminKhan/rag-eval-harness)")


ensure_corpus()
BEST_CFG, BEST = load("configs/best.yaml")
_, BASELINE = load("configs/baseline.yaml")
BEST.retrieve("warm-up")
BASELINE.retrieve("warm-up")


def render_hits(hits) -> str:
    rows = []
    for i, h in enumerate(hits, 1):
        url = f"https://kubernetes.io/docs/{h.chunk.doc_id}/"
        snippet = " ".join(h.chunk.text.split())[:420]
        rows.append(f"**{i}. [{h.chunk.doc_id}]({url})** · score {h.score:.3f}\n\n> {snippet}…")
    return "\n\n".join(rows) or "_No passages found._"


def render_trace(trace: Trace) -> str:
    lines = ["| stage | ms |", "|---|---|"]
    lines += [f"| {s['name']} | {s['duration_ms']:.1f} |" for s in trace.to_dict()]
    return "\n".join(lines)


def ask(question: str):
    question = (question or "").strip()
    if not question:
        return "Type a question about Kubernetes.", "", "", ""
    best_trace, base_trace = Trace(), Trace()
    best_hits = BEST.retrieve(question, best_trace)
    base_hits = BASELINE.retrieve(question, base_trace)
    answer = ""
    if GENERATE:
        from rageval.generation import LLM
        from rageval.generation import answer as generate

        with best_trace.span("generate", model=BEST_CFG.generation.model):
            res = generate(LLM(BEST_CFG.generation), question, best_hits)
        answer = res.text
    return (answer or "_Answer generation is off in this demo; the passages below are what "
            "the model would be given._", render_hits(best_hits), render_trace(best_trace),
            render_hits(base_hits))


with gr.Blocks(title="Kubernetes docs RAG", theme=gr.themes.Soft()) as demo:
    gr.Markdown(header())
    with gr.Row():
        q = gr.Textbox(label="Question", placeholder=EXAMPLES[0], scale=5)
        go = gr.Button("Search", variant="primary", scale=1)
    gr.Examples(EXAMPLES, inputs=q)
    ans = gr.Markdown()
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Best: hybrid + reranker")
            best_out = gr.Markdown()
            trace_out = gr.Markdown()
        with gr.Column():
            gr.Markdown("### Baseline: dense only")
            base_out = gr.Markdown()
    for trigger in (go.click, q.submit):
        trigger(ask, inputs=q, outputs=[ans, best_out, trace_out, base_out])

if __name__ == "__main__":
    demo.launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"))
