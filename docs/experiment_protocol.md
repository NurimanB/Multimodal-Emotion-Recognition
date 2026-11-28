# MELD text experiment protocol

## 1. Problem, task and intended use

Classify the emotion of an English conversational utterance using the target
and up to three preceding turns of the same dialogue. Outputs are neutral,
joy, surprise, anger, sadness, disgust and fear. Intended use is a research
demonstration of contextual text classification. Weighted F1 is the primary
selection metric; macro F1 and class-wise P/R/F1 expose minority-class failures.
This experiment does not validate audio/video, Kazakh/Russian or clinical use.

## 2. Dataset card and preprocessing

- Source: [MELD](https://github.com/declare-lab/MELD), Friends dialogue transcripts;
  official train 9,989, dev 1,109 and test 2,610 utterances. Verify these counts
  in dataset_audit.json for the downloaded files.
- Dataset usage terms are linked by the upstream repository. The project uses
  text CSVs and does not distribute raw media.
- Label order: neutral=0, joy=1, surprise=2, anger=3, sadness=4, disgust=5, fear=6.
- load_split repairs encoding using ftfy, strips surrounding whitespace, maps
  labels and sorts by dialogue/utterance IDs. No stemming/stop-word removal:
  short conversational cues, negation and punctuation can matter.
- Dialogue context stays inside each split and dialogue; no future turns in
  the new experiment. Speaker identities can introduce Friends-specific bias.
- TF-IDF lowercases, retains word unigrams/bigrams and learns IDF on train only.
  The transformer tokenizer produces IDs, attention masks, special tokens,
  dynamic batch padding and at most 256 tokens. Target is first; old context
  is truncated from the left. If the target alone exceeds the budget, context
  is omitted and the target is truncated.
- audit.py writes distributions, repeated text, cross-split exact overlap and
  near-duplicate candidates (character TF-IDF cosine >= .95, length >= 20).
  Review these candidates and disclose overlap. Repeated short stock phrases
  are possible; neither the screen nor official splits prove zero leakage.
  Keep the benchmark fixed rather than deleting test examples after scoring.

## 3. Architecture and rationale

~~~mermaid
flowchart LR
  A[Official MELD CSV] --> B[Repair text / stable labels / sample IDs]
  B --> C[Target and previous turns]
  C --> D[Train-only TF-IDF]
  D --> E[Logistic Regression]
  C --> F[Tokenizer / IDs / mask / padding]
  F --> G[Pretrained transformer encoder]
  G --> H[Seven-class head / cross-entropy]
  E --> I[Dev selection and frozen configuration]
  H --> I
  I --> J[Held-out test / reports / demo]
~~~

TF-IDF + Logistic Regression is a meaningful inexpensive learned baseline.
RoBERTa-large and XLM-RoBERTa-large are historical encoder comparisons.
The new candidate is
[DeBERTa-v3-base](https://huggingface.co/microsoft/deberta-v3-base):
a different English pretrained encoder using disentangled attention and a
different pretraining objective. Its model card distinguishes backbone and
embedding parameters. The report uses sum(p.numel()), not an estimated
parameter count. Superior performance on this task must be measured.

## 4. Training and experiment records

- Optimizer: Trainer default AdamW for the pinned Transformers release.
- Preset: LR 2e-5, batch 4, accumulation 4, effective single-GPU batch 16;
  maximum length 256; maximum 6 epochs; warmup 10%; linear LR schedule;
  weight decay .01; early stopping patience 2 validation evaluations.
- Loss: seven-class cross-entropy, class weights with inverse-frequency
  power .5. Focal loss is disabled in the new preset.
- T4: CUDA mixed precision and gradient checkpointing. Actual GPU, software,
  parameters, peak allocated GPU memory and elapsed training time are recorded.
- Seed 42 by default; optionally predeclare [42, 123, 2025]. Report the number
  actually completed and mean/sample-standard-deviation for multiple seeds.
  A single run has no measured run-to-run standard deviation.
- Best checkpoint: largest dev weighted F1; no checkpoint choice from test.
- trainer_state.json supplies real logged training loss, validation loss/F1
  curves; learning_curves.png is not reconstructed from final scalar metrics.
- Config, training_args.json, checkpoint_manifest.json, source manifest, sample
  IDs, dataset fingerprints and probabilities tie results to source and data.
- Durable full resume needs full optimizer/RNG checkpoints on Drive. Default
  model-only local checkpoints prioritize disk use; an exported best_model
  supports inference but not an exact training resume.

## 5. Evaluation protocol and measured results

TF-IDF vocabulary fits train only; its C/class weights are selected on dev.
The new encoder trains on train and stops/selects on dev. Old checkpoints may
be re-evaluated without retraining. A dev-selected ensemble has fixed weights
and neutral bias before any new test arrays are read. frozen_protocol.json
records the choice and hashes the small configuration/dev artifacts.

Report accuracy, weighted precision/recall/F1, macro F1, test sample count,
per-class P/R/F1, confusion matrices, training time and measured latency.
Weighted F1 reflects frequent classes; macro F1 treats all seven equally.
Neither measures calibrated confidence or generalization to new domains.
Latency is batch 1, preprocessing+forward+postprocessing, with warmup and GPU
synchronization; report p50/p95, repetitions, device and sequence policy.

| Model | Status in source checkout | Test weighted F1 |
|---|---|---:|
| Historical RoBERTa-large | CSV/README attribution; checkpoint recheck needed | 0.6108 |
| Historical XLM-R-large | CSV/README attribution; weights reportedly not preserved | 0.6355 |
| Previously reported ensemble | User-reported; probability artifacts required | not independently measured |
| TF-IDF + Logistic Regression | Completed v3 archive | 0.4166 |
| New DeBERTa-v3-base | Completed seed 42 / 123 / 2025 | 0.6406 / 0.6430 / 0.6467 |
| New dev-selected ensemble | Completed frozen v3; prior test inspection disclosed | 0.6681 |

The two original JSON filenames were mutually reversed relative to CSV/README.
Only the names/content assignment was corrected; all numerical values remain.
See results/legacy_results_provenance.json. The known historical test numbers
have already been inspected; disclose that history instead of claiming the
benchmark has never been seen. Make all new choices on dev.

## 6. Error analysis, limitations and ethics

Use the generated five correct/five incorrect examples and fill in human
hypotheses. Check confident errors, minority emotions, sarcasm, missing prosody,
ambiguous utterances and speaker/context cues. Determine the weakest classes
from the actual report. A confusion matrix cannot reconstruct particular texts.

The examples are selected for class diversity and confidence; do not treat them
as an unbiased estimate of failure rates. Softmax scores are uncalibrated.
Predictions must not be presented as reliable observations of a person's
internal state. Development after test inspection must be documented separately.

## 7. Inference and checkpoint access

README gives exact setup/train/eval/inference commands. The Colab widget displays
raw/cleaned text, target/context, token strings, IDs, mask, sequence length,
emotion and softmax confidence. Use unseen English examples during the demo.
The saved model must contain weights, model config, tokenizer files and the
training run config/label order. Use authorized checkpoint access;
do not put large weights or private tokens in Git.


## References

- https://aclanthology.org/P19-1050/
- https://arxiv.org/abs/2111.09543
- https://huggingface.co/microsoft/deberta-v3-base
