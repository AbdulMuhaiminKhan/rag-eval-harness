"""Web front end: a small JSON API plus a static page.

    python web/server.py                     # http://127.0.0.1:8000
    RAGEVAL_WEB_PORT=9000 python web/server.py

Endpoints
    GET  /api/health          what this server can do (is answer generation reachable?)
    GET  /api/search?q=...    top passages from the best config and the dense baseline, with traces
    POST /api/answer          {"q": "..."} -> an answer from the local LLM over the best config's passages
    GET  /api/results         the committed evaluation results (docs/results/*.json)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from rageval.config import Config  # noqa: E402
from rageval.generation import LLM, answer  # noqa: E402
from rageval.ingest import build_corpus, load_corpus  # noqa: E402
from rageval.retrieval import Hit, Retriever  # noqa: E402
from rageval.tracing import Trace  # noqa: E402

CORPUS = Path(os.environ.get("RAGEVAL_WEB_CORPUS", "data/processed/corpus.jsonl"))
RESULTS = ROOT / "docs" / "results"
STATIC = Path(__file__).resolve().parent / "static"
TOP_K = 5


def _load(config: str) -> tuple[Config, Retriever]:
    cfg = Config.load(config)
    cfg.corpus = str(CORPUS)
    cfg.retrieval.top_k = TOP_K
    return cfg, Retriever(cfg)


if not CORPUS.exists():
    build_corpus(Path("data/raw"), CORPUS.parent,
                 (ROOT / "data" / "CORPUS_COMMIT").read_text(encoding="utf-8").strip())
TITLES = {d.doc_id: d.title for d in load_corpus(CORPUS).values()}
BEST_CFG, BEST = _load("configs/best.yaml")
_, BASELINE = _load("configs/baseline.yaml")
BEST.retrieve("warm-up")
BASELINE.retrieve("warm-up")
LAST_HITS: dict[str, list[Hit]] = {}  # question -> best hits, so /api/answer reuses /api/search

app = FastAPI(title="rageval web")


def _hits_json(hits: list[Hit]) -> list[dict]:
    return [{"rank": i, "chunk_id": h.chunk.chunk_id, "doc_id": h.chunk.doc_id,
             "title": TITLES.get(h.chunk.doc_id, h.chunk.doc_id), "score": round(h.score, 4),
             "url": f"https://kubernetes.io/docs/{h.chunk.doc_id}/", "text": h.chunk.text}
            for i, h in enumerate(hits, 1)]


def _generation_available() -> bool:
    g = BEST_CFG.generation
    if g.provider != "ollama":
        return g.provider == "openai_compatible"
    try:
        tags = requests.get(f"{g.base_url}/api/tags", timeout=1.5).json()
        return any(m.get("name") == g.model for m in tags.get("models", []))
    except (requests.RequestException, ValueError):
        return False


@app.get("/api/health")
def health() -> dict:
    return {"generate": _generation_available(), "model": BEST_CFG.generation.model,
            "docs": len(TITLES), "chunks": len(BEST.chunks)}


@app.get("/api/search")
def search(q: str = Query(..., min_length=3, max_length=500)) -> dict:
    out = {}
    for key, retriever in (("best", BEST), ("baseline", BASELINE)):
        trace = Trace()
        hits = retriever.retrieve(q, trace)
        if key == "best":
            LAST_HITS[q] = hits
            if len(LAST_HITS) > 64:
                LAST_HITS.pop(next(iter(LAST_HITS)))
        out[key] = {"hits": _hits_json(hits), "trace": trace.to_dict()}
    return out


class AnswerRequest(BaseModel):
    q: str


@app.post("/api/answer")
def answer_endpoint(req: AnswerRequest) -> dict:
    if not _generation_available():
        raise HTTPException(503, f"The local model {BEST_CFG.generation.model} isn't running. "
                                 "Start Ollama to enable answers.")
    hits = LAST_HITS.get(req.q) or BEST.retrieve(req.q)
    res = answer(LLM(BEST_CFG.generation), req.q, hits[: BEST_CFG.generation.context_k])
    return {"answer": res.text, "latency_s": round(res.latency_s, 2),
            "tokens_in": res.tokens_in, "tokens_out": res.tokens_out}


@app.get("/api/results")
def results() -> dict:
    def read(name: str):
        return json.loads((RESULTS / name).read_text(encoding="utf-8"))

    return {"ablation": read("ablation.json"), "comparisons": read("comparisons.json"),
            "answers": read("answers.json")}


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


if __name__ == "__main__":
    uvicorn.run(app, host=os.environ.get("RAGEVAL_WEB_HOST", "127.0.0.1"),
                port=int(os.environ.get("RAGEVAL_WEB_PORT", "8000")))
