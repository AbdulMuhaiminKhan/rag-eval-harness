Judged answers: 159 answerable questions where the system gave an answer.

| judge | correct rate [95% CI] | faithful rate [95% CI] |
|---|---|---|
| qwen-v1 | 0.912 [0.87, 0.96] | 0.698 [0.62, 0.77] |
| qwen-v2 | 0.195 [0.14, 0.26] | 0.321 [0.25, 0.39] |
| gemma-v1 | 0.887 [0.84, 0.93] | 0.925 [0.88, 0.96] |

| judges | label | agreement | Cohen's kappa |
|---|---|---|---|
| qwen-v1 vs qwen-v2 | correct | 0.283 | 0.046 |
| qwen-v1 vs qwen-v2 | faithful | 0.585 | 0.273 |
| qwen-v1 vs gemma-v1 | correct | 0.912 | 0.514 |
| qwen-v1 vs gemma-v1 | faithful | 0.736 | 0.204 |
| qwen-v2 vs gemma-v1 | correct | 0.296 | 0.043 |
| qwen-v2 vs gemma-v1 | faithful | 0.396 | 0.074 |
