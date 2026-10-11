# RAG with Evaluation Harness

A retrieval-augmented question answering pipeline over the Kubernetes documentation, built
around an evaluation harness that measures every design choice instead of assuming it helps.
Everything runs locally for $0: open-source embedders and rerankers via sentence-transformers,
Chroma as the vector store, and a local LLM through Ollama (any OpenAI-compatible endpoint can be
configured instead).

## What it measures

- **Retrieval:** Recall@k, MRR@10 and nDCG@10 (k = 1, 3, 5, 10), plus document-level recall.
- **Answers:** correctness and faithfulness from an LLM judge, abstention rate on unanswerable
  questions, and false abstentions on answerable ones.
- **Judge calibration:** agreement (accuracy, Cohen's kappa) between the LLM judge and a
  hand-labelled subset.
- **Cost and speed:** p50/p95 retrieval and generation latency, tokens and USD per query.
- **Uncertainty:** every headline number is reported with a 95% bootstrap confidence interval
  over questions; comparisons between configs use a paired bootstrap.

## How labels work

Gold labels are **evidence spans**: a `doc_id` plus a verbatim quote from the cleaned page, not
chunk ids. A retrieved chunk counts as relevant when it overlaps a gold span by at least half of
the shorter of the two, and each gold span is credited once. This means the same eval set scores
fixed-size chunks, semantic chunks, or any future chunker fairly.

```json
{"qid": "h0001", "question": "What is the maximum size of data a ConfigMap can hold?",
 "type": "factoid", "answerable": true,
 "evidence": [{"doc_id": "concepts/configuration/configmap",
               "quote": "The data stored in a\nConfigMap cannot exceed 1 MiB."}],
 "reference_answer": "1 MiB.", "origin": "hand", "reviewed": true}
```

Question types: `factoid`, `procedural`, `conceptual`, `comparison`, `multi_hop`, `unanswerable`.

## Ablation grid

`configs/ablation.yaml` is the full grid of chunking (fixed, semantic) x retriever (dense, BM25,
hybrid with reciprocal rank fusion) x reranker (none, cross-encoder) x embedding model
(all-MiniLM-L6-v2, bge-small-en-v1.5).

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate      # source .venv/bin/activate on Linux/macOS
pip install -e ".[dev,report]"
ollama pull qwen2.5:7b                              # only needed for answers and the judge

rageval ingest --ref "$(cat data/CORPUS_COMMIT)"    # Kubernetes docs at a pinned commit
rageval validate                                    # every evidence quote must resolve
rageval eval --config configs/baseline.yaml         # retrieval metrics
rageval eval --config configs/baseline.yaml --generate   # + answers and LLM judge
rageval ablate --grid configs/ablation.yaml
rageval ask "How do I roll back a Deployment?"
```

Growing the eval set:

```bash
rageval gen-questions --n 50      # LLM drafts into data/eval/candidates.jsonl
rageval review                    # accept / edit / reject each draft by hand
```

## CI

On every pull request, GitHub Actions runs the unit tests, then evaluates `configs/ci.yaml` on a
seeded slice of the corpus (all gold documents plus 300 random distractors) and fails if
Recall@5, MRR@10 or nDCG@10 drops more than 0.03 below `ci/baseline.json`.

## Results

No results are reported yet. This section will only ever contain numbers produced by
`rageval eval` / `rageval ablate` on the committed eval set, with confidence intervals.

## Layout

```
src/rageval/
  ingest.py      download + clean Kubernetes docs, pinned by commit
  chunking.py    fixed and semantic chunkers, offset-preserving
  retrieval.py   Chroma dense, BM25, hybrid RRF, cross-encoder rerank
  generation.py  Ollama / OpenAI-compatible LLM client, answer prompt
  judge.py       LLM-as-judge and judge/human agreement
  dataset.py     eval set schema and evidence span resolution
  metrics.py     span-based retrieval metrics, bootstrap CIs
  evaluate.py    run a config over the eval set, write per-question rows + summary
  qgen.py        draft questions for human review
  cli.py         `rageval` command
configs/         baseline, CI and ablation configs
data/eval/       eval questions (committed)
```
