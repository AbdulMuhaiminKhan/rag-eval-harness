| change | Δ recall@5 [95% CI] | Δ mrr@10 [95% CI] |
|---|---|---|
| cross-encoder reranker, on dense | +0.187 [+0.123, +0.260] | +0.200 [+0.144, +0.258] |
| cross-encoder reranker, on hybrid | +0.082 [+0.026, +0.140] | +0.078 [+0.027, +0.129] |
| hybrid instead of dense | +0.123 [+0.070, +0.181] | +0.148 [+0.111, +0.192] |
| hybrid instead of BM25 | -0.012 [-0.064, +0.041] | +0.033 [-0.020, +0.080] |
| bge-small instead of MiniLM, dense | +0.053 [-0.018, +0.123] | +0.071 [+0.017, +0.125] |
| semantic instead of fixed chunks, dense | +0.047 [-0.018, +0.117] | +0.012 [-0.041, +0.066] |
| semantic instead of fixed chunks, hybrid bge | +0.058 [+0.003, +0.117] | +0.052 [-0.005, +0.106] |
| semantic instead of fixed chunks, best config | -0.015 [-0.073, +0.041] | +0.003 [-0.047, +0.053] |
