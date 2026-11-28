# Exploratory ensemble refinement v2

No neural model was retrained. This experiment recombines the two existing
probability matrices supplied in the evidence archive. The original results
remain unchanged in [the source ZIP](../evidence/run_v1_evidence.zip).

## Protocol

- Dataset: the same 1,109 dev and 2,610 test examples, with identical sample IDs,
  gold labels and fingerprints independently checked against canonical MELD CSVs.
- Search: 21 first-model weights (0..1 by .05) x 31 neutral biases
  (-.5..1 by .05), 651 candidates in total, evaluated on dev only.
- Tie-break: nearest original configuration, then smaller absolute bias,
  then smaller first-model weight.
- Selected and frozen BEFORE test evaluation:
  RoBERTa 0.6, DeBERTa 0.4,
  neutral log-score bias 0.75.
- [selection.json](selection.json) and its SHA-256 record the frozen choice.
- The original test result had already been inspected. This is a documented
  exploratory iteration, not a new independent holdout experiment.
- No alternative candidate was selected using test results.

## Results

| Version | Dev weighted F1 | Test weighted F1 | Test macro F1 | Test accuracy |
|---|---:|---:|---:|---:|
| Original v1 | 0.664584 | 0.657557 | 0.509334 | 0.660536 |
| Dev-refined v2 | 0.666328 | 0.658473 | 0.511167 | 0.662835 |

The test weighted-F1 difference is +0.000916,
or +0.0916 percentage points.
Correct classifications increased from
1724 to
1730 out of 2,610.

A paired dialogue-block percentile bootstrap (2,000 resamples, fixed seed 2025,
280 dialogues) gives a 95% interval for the F1 difference of
[-0.003226, +0.004886].
It includes zero: the small observed improvement does not establish a reliable
generalization gain. This interval also does not undo earlier test inspection.

The previously reported 0.664 test score was not reproduced or surpassed by this
iteration. Historical aggregate JSONs alone cannot reconstruct XLM-R probabilities.

## Files

- comparison.csv: both versions on the same splits.
- dev_search.csv and dev_search_heatmap.png: complete dev search, without test values.
- per_class_comparison.csv: class-wise trade-offs.
- report/: confusion matrices, P/R/F1 and real diagnostic examples.
- evaluation.json: full metrics and uncertainty.
- provenance.json: source archive, frozen-selection and module hashes.

## Reproduce without a GPU

From the repository, after installing requirements-experiments.txt and the package:

~~~bash
python -m mer.evaluation.refine select --archive results/experiments/evidence/run_v1_evidence.zip --output-dir outputs/refinement_reproduction
python -m mer.evaluation.refine evaluate --archive results/experiments/evidence/run_v1_evidence.zip --output-dir outputs/refinement_reproduction
python -m mer.evaluation.report --run-dir outputs/refinement_reproduction
~~~

Optionally pass --data-dir pointing to the official MELD CSVs for an independent
dataset fingerprint check. Our recorded experiment used this check.
The uploaded source archive is unchanged. Model weights are not required.
