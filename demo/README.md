# Demo

A small Gradio app that runs the best retrieval setup (hybrid BM25 + MiniLM, cross-encoder
rerank) next to the dense-only baseline on any question, with per-stage timings.

```bash
pip install -e . "gradio>=5,<6"
python demo/app.py                          # http://127.0.0.1:7860
RAGEVAL_DEMO_GENERATE=1 python demo/app.py  # also generate an answer with Ollama
```

On first start it builds the corpus at the pinned docs commit and the indexes (a few minutes on
CPU, cached afterwards).
