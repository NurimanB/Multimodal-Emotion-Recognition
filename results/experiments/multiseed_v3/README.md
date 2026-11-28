# Three-seed ensemble experiment

Checked metrics and figures derived from the original v3 run archive.
comparison.csv includes all seven evaluated predictors on dev and test.
test_per_class.csv and review_examples.csv expose class-wise results and
five correct / five incorrect cases with restored dialogue context.

The selected ensemble is 0.35 RoBERTa + 0.65 equal mean of DeBERTa seeds
42, 123, 2025, with neutral log-score offset +0.85. Test weighted F1:
0.6680791110593878; macro F1: 0.5112797606007202.

uncertainty.json contains conditional dialogue-block bootstrap intervals.
The protocol explicitly acknowledges earlier benchmark test inspection.
Source: ../evidence/multiseed_v3_evidence.zip (unchanged archive).
