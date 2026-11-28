# Measured final experiment addendum

This supplements docs/experiment_protocol.md with the completed v3 results.

Official MELD splits: train 9,989 / dev 1,109 / test 2,610.
Primary task is English text emotion classification, seven labels, three prior
turns, no future turns. V3 is an exploratory continuation after earlier test
inspection; all within-v3 checkpoint/blend choices used dev.

Completed DeBERTa seeds: 42, 123, 2025, with the identical learning configuration.
Individual test weighted F1 mean ± sample SD: 0.643441 ± 0.003055.
Equal probability averaging: 0.658646 test weighted F1.
Final ensemble: 0.35 RoBERTa + 0.65 equal DeBERTa mean, neutral log-score
offset +0.85, all selected from 651 fixed dev candidates and frozen before
new test evaluation.

Final test: weighted F1 0.668079, macro F1 0.511280,
accuracy 0.677395, weighted precision 0.666774,
weighted recall 0.677395. Full checked table:
results/experiments/multiseed_v3/comparison.csv. Per-class metrics:
results/experiments/multiseed_v3/test_per_class.csv. Ten reviewed cases:
results/experiments/multiseed_v3/review_examples.csv.

Main weakness: fear F1 0.2881 and disgust F1 0.3115. Several selected errors
have high neutral scores; confidence is uncalibrated. Friends/domain/speaker
bias and missing vocal/visual information limit the conclusions.
No no-context ablation, new XLM-R run, external-domain validation, or working
multimodal fusion is claimed.

Training used T4 FP16 and gradient checkpointing. Each DeBERTa has 184,427,527
trainable parameters, peak allocated training memory about 3.15 GiB, and
training times 42.50 / 42.67 / 50.77 min. The final four-checkpoint demo
has 908,649,500 parameters; measured latency/throughput and exact scope are
in results/experiments/inference_demo/evidence.json.

Conditional 95% dialogue-block-bootstrap interval:
[0.6465035084136082, 0.6905201468914071]. This does not erase prior benchmark
inspection. Fresh external dialogues are the next validation requirement.

Evidence: results/experiments/evidence/multiseed_v3_evidence.zip,
reproducible notebooks, run configurations and source links in README.md.
