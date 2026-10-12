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

## Calibration against reference labels

40 judged answers were sampled (`scripts/make_label_sheet.py`, seed 0). Claude suggested each
label by reading the question, reference answer, system answer and all five retrieved passages;
the author then reviewed every suggestion on a labelling page and confirmed all 40 without
changes. The reasons for every negative label are in `data/eval/human_labels.jsonl`. The reference labels
37/40 answers correct and 38/40 faithful.

| judge | label | accuracy | Cohen's kappa | judge positive rate |
|---|---|---|---|---|
| qwen-v1 | correct | 0.900 | 0.279 | 0.925 |
| qwen-v1 | faithful | 0.650 | -0.094 | 0.700 |
| qwen-v2 | correct | 0.275 | 0.040 | 0.200 |
| qwen-v2 | faithful | 0.400 | 0.055 | 0.350 |
| gemma-v1 | correct | 0.875 | 0.383 | 0.850 |
| gemma-v1 | faithful | 0.925 | 0.362 | 0.925 |

With only 2 or 3 negative labels in the sample, kappa is noisy; the accuracies are the more
stable signal. gemma2 9B is the only judge that agrees with the reference on both labels. The
qwen judge's faithfulness scores agree no better than chance, and the stricter qwen prompt fails
most correct answers.
