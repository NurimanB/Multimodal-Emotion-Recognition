# Emotion recognition — experimental results

Completed text-only English MELD experiments. The official test split contains
2,610 utterances with the fixed label order: neutral, joy, surprise, anger,
sadness, disgust, fear.

## Latest completed runs

| Run | Dev weighted F1 | Test weighted F1 | Test macro F1 |
|---|---:|---:|---:|
| TF-IDF + Logistic Regression | 0.418959 | 0.416621 | 0.248710 |
| RoBERTa current recheck | 0.637366 | 0.613232 | 0.476107 |
| DeBERTa seed 42 | 0.632412 | 0.640631 | 0.469680 |
| DeBERTa seed 123 | 0.631273 | 0.642999 | 0.493418 |
| DeBERTa seed 2025 | 0.620923 | 0.646693 | 0.492233 |
| Equal three-seed probability mean | 0.639328 | 0.658646 | 0.506526 |
| Final frozen ensemble | 0.668962 | 0.668079 | 0.511280 |

The final ensemble uses 0.35 RoBERTa + 0.65 DeBERTa mean, then adds 0.85 to
the neutral log-score and renormalizes. All within-v3 choices used dev; earlier
test results had already been inspected. No external-domain validation is claimed.

Individual DeBERTa test F1 mean ± sample SD is 0.643441 ± 0.003055.
This differs from F1 of the probability-averaged predictor (0.658646).

## Files

- [Full comparison](experiments/multiseed_v3/comparison.csv)
- [Class-wise results](experiments/multiseed_v3/test_per_class.csv)
- [Ten reviewed examples and context](experiments/multiseed_v3/review_examples.csv)
- [Figures](experiments/multiseed_v3/figures)
- [Recorded ensemble inference and latency](experiments/inference_demo/evidence.json)
- [Final experiment analysis](../docs/final_results.md)
- [Two-model refinement](experiments/ensemble_refinement_v2/README.md)
- [Raw v1 run archive](experiments/evidence/run_v1_evidence.zip)
- [Raw multiseed v3 archive](experiments/evidence/multiseed_v3_evidence.zip)

Archives preserve the original recorded configurations, paths, predictions,
trainer logs and freeze records. Their contents have not been rewritten.
Large model weights are stored separately on Drive.

## Historical encoder runs

| Metric | XLM-RoBERTa-large | RoBERTa-large |
|---|---:|---:|
| Dev weighted F1 | 0.6126 | 0.6374 |
| Test weighted F1 | 0.6355 | 0.6108 |
| Test macro F1 | 0.4806 | 0.4727 |
| Test accuracy | 0.6215 | 0.5996 |
| Training time, min | 90.4 | 79.2 |

Source: model_comparison.csv and the two legacy result JSONs. Their original
filenames had been swapped; see legacy_results_provenance.json.
The current RoBERTa recheck is a separate scoring record. XLM-R weights and
per-example probabilities were unavailable for the current ensemble.

Saved RoBERTa checkpoint:
MyDrive/diploma-mer/outputs/text_roberta_large_ctx3/saved_model.
The three DeBERTa best_model directories remain with their recorded Drive runs;
the inference script locates those run folders from metadata.
