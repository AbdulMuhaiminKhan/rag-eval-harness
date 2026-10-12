# Web front end

A single-page front end over a small FastAPI server.

```bash
pip install -e ".[web]"
python web/server.py        # http://127.0.0.1:8000
```

- **Search** runs the best configuration (hybrid BM25 + MiniLM, cross-encoder rerank) and the
  dense-only baseline on the same question. It shows each list's top 5 passages with the query
  terms highlighted, marks passages only one setup found, and draws per-stage timings. With
  Ollama running and `qwen2.5:7b` pulled, **Generate answer** writes an answer from the best
  setup's passages. Link to a question with `/?q=...`.
- **Results** draws the committed evaluation: headline numbers with CIs, all 24 configurations
  (quality against latency), one-change-at-a-time paired comparisons, and the judge calibration.
  It reads `docs/results/*.json`, so it always matches the repo.

API: `GET /api/health`, `GET /api/search?q=`, `POST /api/answer {"q": ...}`, `GET /api/results`.
