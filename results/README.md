# Emotion recognition — experimental results

Text-only English MELD classification. Train: 9,989; dev: 1,109; test: 2,610.
All new within-run checkpoint and blend choices use dev. The benchmark had
already been inspected in prior experiments.

## Current results

| Run | Dev weighted F1 | Test weighted F1 | Test macro F1 |
|---|---:|---:|---:|
| TF-IDF + Logistic Regression | 0.418959 | 0.416621 | 0.248710 |
| RoBERTa current recheck | 0.637366 | 0.613232 | 0.476107 |
| DeBERTa seed 42 | 0.632412 | 0.640631 | 0.469680 |
| DeBERTa seed 123 | 0.631273 | 0.642999 | 0.493418 |
| DeBERTa seed 2025 | 0.620923 | 0.646693 | 0.492233 |
| Equal DeBERTa seed mean | 0.639328 | 0.658646 | 0.506526 |
| RoBERTa + DeBERTa seed mean | 0.668962 | 0.668079 | 0.511280 |

The final ensemble uses 0.35 RoBERTa + 0.65 equal DeBERTa probability mean,
with neutral log-score offset +0.85, selected on dev.
Its test accuracy is 0.677395. Individual DeBERTa test F1 mean ± sample SD
is 0.643441 ± 0.003055; this differs from the probability-mean predictor.

## Files

- One *_result.json per model, seed or ensemble, including per-class metrics
  and confusion matrices, matching the existing encoder result layout.
- [model_comparison.csv](model_comparison.csv): all historical and current runs.
- [error_examples.csv](error_examples.csv): five correct / five incorrect
  ensemble predictions, context and human error hypotheses.
- figures/: class distribution, learning curves, final confusion matrix and
  test comparison, saved once as PNG.
- [legacy_results_provenance.json](legacy_results_provenance.json): attribution
  correction for the two older JSON files.

Earlier two-model ensemble iterations are retained in separate v1/v2 JSONs.
Full raw run exports, probabilities, training logs and weights are retained
outside the repository. The notebook and scripts write full outputs to Drive.

## Historical encoder runs

| Metric | XLM-RoBERTa-large | RoBERTa-large |
|---|---:|---:|
| Dev weighted F1 | 0.6126 | 0.6374 |
| Test weighted F1 | 0.6355 | 0.6108 |
| Test macro F1 | 0.4806 | 0.4727 |
| Test accuracy | 0.6215 | 0.5996 |
| Training minutes | 90.4 | 79.2 |

The current RoBERTa recheck is a separate scoring record.
XLM-R weights/probabilities were unavailable for the new ensemble.

Rare emotions remain weak: final ensemble fear F1 0.2881 and disgust F1 0.3115.
Scores are uncalibrated and external-domain performance has not been evaluated.
