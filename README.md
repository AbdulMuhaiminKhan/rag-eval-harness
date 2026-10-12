# RAG with Evaluation Harness

A retrieval-augmented question answering pipeline over the Kubernetes documentation, built
around an evaluation harness that measures every design choice instead of assuming it helps.
Everything runs locally for $0: open-source embedders and a cross-encoder reranker via
sentence-transformers, Chroma as the vector store, BM25, and a local LLM through Ollama (any
OpenAI-compatible endpoint can be configured instead).

## Results

All numbers below were produced by `rageval` on the committed eval set (186 questions, 171
answerable) over the Kubernetes docs at commit `35aa173` (1,269 pages, v1.37). Brackets are 95%
bootstrap confidence intervals over questions. Full tables: [docs/results](docs/results).

### Retrieval: 24-config ablation

| config | Recall@5 | MRR@10 | p50 latency |
|---|---|---|---|
| dense, MiniLM (baseline) | 0.582 [0.51, 0.65] | 0.430 [0.37, 0.49] | 16 ms |
| BM25 | 0.716 [0.65, 0.78] | 0.546 [0.49, 0.61] | 53 ms |
| hybrid (dense + BM25, RRF) | 0.705 [0.64, 0.77] | 0.578 [0.52, 0.64] | 64 ms |
| dense + cross-encoder rerank | 0.769 [0.71, 0.83] | 0.630 [0.57, 0.69] | 159 ms |
| **hybrid + cross-encoder rerank** | **0.787 [0.73, 0.85]** | **0.657 [0.60, 0.72]** | 219 ms |

Fixed 1,000-character chunks with 150 overlap; latency measured on an RTX 4070 laptop GPU.
All 24 configs are in [docs/results/ablation.md](docs/results/ablation.md).

![Retrieval quality vs latency](docs/results/quality_vs_latency.png)

What changing one thing at a time does (paired bootstrap, Recall@5):

| change | Δ Recall@5 [95% CI] |
|---|---|
| add the cross-encoder reranker to dense | +0.187 [+0.123, +0.260] |
| add the cross-encoder reranker to hybrid | +0.082 [+0.026, +0.140] |
| hybrid instead of dense | +0.123 [+0.070, +0.181] |
| hybrid instead of BM25 | −0.012 [−0.064, +0.041] |
| bge-small instead of MiniLM (dense) | +0.053 [−0.018, +0.123] |
| semantic instead of fixed chunks (best config) | −0.015 [−0.073, +0.041] |

Takeaways:

- **The reranker is the biggest single win**, and it is worth it even on top of hybrid search,
  at roughly 150 ms extra per query on a GPU (about 2 s on CPU).
- **BM25 alone is a strong baseline** on technical docs full of exact identifiers
  (`maxUnavailable`, `kubectl drain`); fusing it with dense retrieval mostly helps ranking (MRR).
- **Semantic chunking and a better small embedder did not make a reliable difference** once a
  reranker was in place: their intervals cross zero.

Where the best config still fails (19 of 171 answerable questions miss the evidence in the top 10):
14 retrieve the right page but the wrong passage, 5 the wrong page. Comparison and multi-hop
questions are hardest (Recall@5 0.71 and 0.61 vs 0.84 for factoid). See the failure table in
[docs/results/ablation.md](docs/results/ablation.md).

### Answers

The best retrieval config feeds its top 5 chunks to `qwen2.5:7b` through Ollama
(`configs/best.yaml`).

| measure | result |
|---|---|
| answers correct (gemma2 9B judge, all 171 answerable) | 0.825 [0.77, 0.88] |
| answers faithful to the passages (same judge) | 0.930 [0.89, 0.97] |
| abstains on unanswerable questions | 15 of 15 |
| abstains on answerable questions (false abstention) | 0.070 [0.035, 0.111] |
| generation latency, p50 / p95 | 4.1 s / 7.7 s |
| prompt / completion tokens per query | ~1,190 / ~83 |
| cost per query | $0 (local model) |

Correctness and faithfulness come from the one judge that held up in calibration. Abstaining
on an answerable question counts as not correct. Picking the judge took most of the work.
Three judges graded the same 159 answers (the system's answer to every answerable question
where it did not abstain):

| judge | correct rate [95% CI] | faithful rate [95% CI] |
|---|---|---|
| qwen2.5 7B, prompt v1 | 0.912 [0.87, 0.96] | 0.698 [0.62, 0.77] |
| qwen2.5 7B, prompt v2 (stricter definitions) | 0.195 [0.14, 0.26] | 0.321 [0.25, 0.39] |
| gemma2 9B, prompt v1 | 0.887 [0.84, 0.93] | 0.925 [0.88, 0.96] |

The two qwen prompts agree on correctness only at chance level (Cohen's kappa 0.05). The v2
prompt fails plainly right answers, such as "10 old ReplicaSets by default", for not naming the
config field. Even the two v1 judges agree only moderately (kappa 0.51 on correctness, 0.20 on
faithfulness).

To choose between them, 40 sampled answers were graded against all five retrieved passages.
At the project author's request, Claude graded them as a reference grader, so this is not a
human-agreement figure. The reasons for each negative label are in
`data/eval/human_labels.jsonl`.

| judge | correct: accuracy / kappa | faithful: accuracy / kappa |
|---|---|---|
| qwen2.5 7B, prompt v1 | 0.900 / 0.28 | 0.650 / −0.09 |
| qwen2.5 7B, prompt v2 | 0.275 / 0.04 | 0.400 / 0.06 |
| **gemma2 9B, prompt v1** | **0.875 / 0.38** | **0.925 / 0.36** |

gemma2 9B is the only judge that agrees with the reference on both labels. The qwen judge's
faithfulness scores agree no better than chance, which also argues against letting the model
that wrote the answers grade them. The reference marked only 3 of 40 answers incorrect and 2
unfaithful, so the kappas are noisy. Treat the headline correctness as about ±0.05, plus some
judge error. [docs/results/judges.md](docs/results/judges.md) has every pairing.

## Caveats

- **LLM-drafted questions are easier than hand-written ones.** The dense baseline scores 0.63
  Recall@5 on the 126 LLM-drafted questions vs 0.46 on the 45 hand-written answerable ones; the
  best config scores 0.80 vs 0.74. Drafts are written while looking at the passage and share its
  wording, which flatters lexical and embedding retrieval alike. Treat absolute numbers as
  optimistic; the comparisons between configs are the robust part.
- The eval set was drafted and reviewed with help from Claude; review decisions are logged in
  `data/eval/review_log.jsonl` and each question records who reviewed it.
- 186 questions give CIs about ±0.06 wide on Recall@5: enough to see the reranker and hybrid
  effects, not enough to rank the top configs against each other.
- GPU and CPU embeddings rank a few near-ties differently (Recall@5 moved by about 0.015 on the
  CI slice), so CI baselines are generated on CPU, like the CI runner.

## How it is evaluated

**Retrieval:** Recall@k, MRR@10, nDCG@10 (k = 1, 3, 5, 10) and document-level recall.
**Answers:** correctness and faithfulness from an LLM judge, abstention on unanswerable questions
and false abstentions on answerable ones. **Judge calibration:** agreement and Cohen's kappa
against a labelled reference sample. **Cost and speed:** p50/p95 latency, tokens and USD per query.
Every headline number has a 95% bootstrap CI; comparisons use a paired bootstrap.

### Labels are evidence spans, not chunk ids

Each answerable question cites a `doc_id` plus a verbatim quote from the cleaned page. A
retrieved chunk counts as relevant when it overlaps a gold span by at least half of the shorter
of the two, and each span is credited once, so the same labels score any chunking strategy.
When the same fact is stated on another page, it is listed as an alternative.

```json
{"qid": "h0001", "question": "What is the maximum size of data a ConfigMap can hold?",
 "type": "factoid", "answerable": true,
 "evidence": [{"doc_id": "concepts/configuration/configmap",
               "quote": "The data stored in a\nConfigMap cannot exceed 1 MiB."}],
 "reference_answer": "1 MiB.", "origin": "hand", "reviewed": true}
```

| type | factoid | procedural | conceptual | comparison | multi_hop | unanswerable |
|---|---|---|---|---|---|---|
| questions | 37 | 60 | 48 | 17 | 9 | 15 |

### Building the eval set

60 questions are hand-written, including all multi-hop and unanswerable ones. The
rest were drafted by a local LLM from sampled passages and then reviewed: 126 of 171 drafts were
accepted. The drafting step automatically rejects quotes that are not in the page, questions
that copy the passage's wording, near-duplicates, and boilerplate that repeats across many pages
(kubectl global flags, API list parameters), which would make a single gold page meaningless.

## CI

Every push and pull request runs lint and unit tests, then evaluates `configs/ci.yaml` on a
seeded slice of the corpus (every gold page plus 300 random distractors, about 5,800 chunks) and
fails if Recall@5, MRR@10 or nDCG@10 drops more than 0.03 below `ci/baseline.json`. Each run
records a fingerprint of the eval set, corpus and pipeline config; if the baseline was made
from a different one, the gate stops with "stale baseline" instead of reporting a fake
regression. Regenerate it with `python scripts/update_ci_baseline.py`.

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate      # source .venv/bin/activate on Linux/macOS
pip install -e ".[dev,report]"
ollama pull qwen2.5:7b                              # only needed for answers and the judge

rageval ingest --ref "$(cat data/CORPUS_COMMIT)"    # Kubernetes docs at the pinned commit
rageval validate                                    # every evidence quote must resolve
rageval eval --config configs/best.yaml             # retrieval metrics
rageval eval --config configs/best.yaml --generate --out runs/answers   # + answers and judge
rageval ablate --grid configs/ablation.yaml --out runs/ablation
python scripts/report.py runs/ablation              # results table, failures, chart
python scripts/compare.py runs/ablation             # one-change-at-a-time comparisons
rageval ask "How do I roll back a Deployment?" --config configs/best.yaml

# judge comparison and calibration against reference labels
rageval rejudge --run runs/answers --out runs/answers_gemma --model gemma2:9b
python scripts/judge_agreement.py qwen=runs/answers gemma=runs/answers_gemma
rageval calibrate --run runs/answers --run runs/answers_gemma
```

Growing the eval set:

```bash
rageval gen-questions --n 50                 # LLM drafts into data/eval/candidates.jsonl
rageval review --reviewer "Your Name"        # accept / edit / reject each draft
```

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
  qgen.py        draft questions for review, with quality filters
  cli.py         the `rageval` command
configs/         baseline, best, CI and ablation configs
data/eval/       eval questions and review log (committed)
docs/results/    measured results
scripts/         reporting, comparisons, review and CI baseline helpers
```
