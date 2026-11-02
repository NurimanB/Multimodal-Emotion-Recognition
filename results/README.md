
# Emotion Recognition — Experimental Results

KBTU diploma project (6B06103 Mathematical and Computer Modeling): evaluation and comparison of transformer-based emotion recognition models on the MELD dataset.

This folder contains the results of our first text-based emotion recognition experiments.

---

## Current stage: Text emotion recognition on MELD

- **Dataset:** MELD (train 9,989 / dev 1,109 / test 2,610 utterances).
- **Emotion classes:** neutral, joy, surprise, anger, sadness, disgust, fear.
- **Main model:** `xlm-roberta-large`
- **Reference model:** `roberta-large`
- **Context:** 3 previous dialogue turns with speaker names.
- **Training:** fine-tuning with class-weighted loss.
- **Metrics:** accuracy, weighted F1, macro F1, per-class report and confusion matrix.

---

## Model comparison

Both models were fine-tuned and evaluated on the same MELD dataset.

| Metric | XLM-RoBERTa-large | RoBERTa-large |
|---|---:|---:|
| Dev weighted F1 | 0.6126 | **0.6374** |
| Test weighted F1 | **0.6355** | 0.6108 |
| Test macro F1 | **0.4806** | 0.4727 |
| Test accuracy | **0.6215** | 0.5996 |
| Training time | 90.4 min | 79.2 min |

### Evaluation metrics

- **Accuracy:** proportion of correctly classified utterances.
- **Weighted F1:** F1-score weighted by class frequency.
- **Macro F1:** average F1-score across all seven classes.
- **Precision:** proportion of correct positive predictions.
- **Recall:** proportion of actual instances correctly identified.
- **Confusion matrix:** shows which emotions are frequently confused.

---

## Key findings

- **XLM-RoBERTa-large** achieved the highest test weighted F1 of **0.6355** and accuracy of **62.15%**.
- **RoBERTa-large** achieved higher validation weighted F1 of **0.6374**.
- XLM-RoBERTa-large showed slightly better test performance.
- Performance varies significantly across emotion classes.
- Rare emotions such as fear and disgust remain challenging.
- Further experiments are needed to establish consistent superiority.

---

## Results layout

```text
results/
├── README.md
├── RoBERTa-large_result.json
├── XLM-RoBERTa-large_result.json
└── model_comparison.csv
```

### File descriptions

- `README.md` — experimental results documentation.
- `RoBERTa-large_result.json` — RoBERTa-large evaluation metrics.
- `XLM-RoBERTa-large_result.json` — XLM-RoBERTa-large evaluation metrics.
- `model_comparison.csv` — summary of model performance.

The JSON files contain detailed classification reports, evaluation metrics and confusion matrices.

---

## Saved models

The trained RoBERTa-large checkpoint is stored separately on Google Drive.

**Google Drive location:**

```text
MyDrive/
└── diploma-mer/
    └── outputs/
        └── text_roberta_large_ctx3/
            └── saved_model/
                ├── model.safetensors
                ├── config.json
                ├── tokenizer.json
                ├── tokenizer_config.json
                ├── trainer_state.json
                └── training_args.bin
```

### Saved checkpoint information

- **Architecture:** RoBERTa-large
- **Task:** seven-class emotion classification
- **Dataset:** MELD
- **Context:** 3 previous dialogue turns
- **Checkpoint:** 3125 training steps
- **Model weights size:** approximately 1.32 GB
- **Storage:** Google Drive

The saved checkpoint has not yet been independently re-evaluated against the reported final metrics.

XLM-RoBERTa-large evaluation results are available, but its trained weights have not been preserved.

Large model files are excluded from GitHub due to their size.

---

## Reproducibility

The repository contains the code and configurations required to reproduce the experiments.

```text
configs/                        Training configurations
notebooks/                      Google Colab notebook
scripts/download_meld.py        MELD dataset downloader
src/mer/data/meld.py            Data preprocessing
src/mer/text/train.py           Model training and evaluation
tests/test_meld.py              Unit tests
```

### Training workflow

1. Download and prepare the MELD dataset.
2. Add dialogue context to each utterance.
3. Tokenize text using the corresponding model tokenizer.
4. Fine-tune the pretrained transformer.
5. Evaluate on validation and test datasets.
6. Calculate classification metrics.
7. Save experimental results to Google Drive.

---

## Next steps

1. **Context comparison:** evaluate models with and without previous dialogue turns.
2. **Error analysis:** investigate confusion matrices and misclassified emotions.
3. **Model improvements:** improve recognition of underrepresented emotions.
4. **Demo application:** implement emotion prediction for custom text.
5. **Multimodal extension:** integrate audio and video emotion recognition.

---

## Conclusion

We successfully completed the initial training and evaluation experiments using **XLM-RoBERTa-large** and **RoBERTa-large**.

XLM-RoBERTa-large achieved the highest test performance in the current experiments, while RoBERTa-large achieved better validation weighted F1.

These experiments establish a foundation for further development of context-aware and multimodal human emotion recognition.
