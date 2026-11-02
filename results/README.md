
# Emotion Recognition — Experimental Results

KBTU diploma project (6B06103 Mathematical and Computer Modeling): evaluation of transformer-based emotion recognition models on the MELD dataset.

This folder contains the results of our first text-based emotion recognition experiments.

## Current stage: Text emotion recognition on MELD

- **Dataset:** MELD (9,989 train / 1,109 dev / 2,610 test utterances).
- **Classes:** neutral, joy, surprise, anger, sadness, disgust, fear.
- **Main model:** `xlm-roberta-large`.
- **Reference model:** `roberta-large`.
- **Context:** up to 3 previous dialogue turns with speaker names.
- **Training:** fine-tuning with class-weighted loss.
- **Metrics:** accuracy, weighted F1, macro F1, per-class classification report and confusion matrix.

## Model comparison

Both models were evaluated on the same MELD test set.

| Metric | XLM-RoBERTa-large | RoBERTa-large |
|---|---:|---:|
| Dev weighted F1 | 0.6126 | 0.6374 |
| Test weighted F1 | **0.6355** | 0.6108 |
| Test macro F1 | **0.4806** | 0.4727 |
| Test accuracy | **0.6215** | 0.5996 |
| Training time | 90.4 min | 79.2 min |

## Evaluation metrics

- **Accuracy:** proportion of correctly classified utterances.
- **Weighted F1:** F1-score averaged according to class frequency.
- **Macro F1:** average F1-score across all seven emotions, treating every class equally.
- **Precision:** proportion of predicted instances of a class that are correct.
- **Recall:** proportion of actual instances of a class correctly identified.

## Key findings

- XLM-RoBERTa-large achieved the highest test weighted F1 of **0.6355**.
- RoBERTa-large achieved a higher validation weighted F1 of **0.6374**.
- The difference between weighted and macro F1 indicates uneven performance across emotion classes.
- Rare emotions, particularly fear and disgust, remain challenging.
- Further experiments are required to establish whether one architecture consistently outperforms the other.

## Results layout

```text
results/
├── README.md
├── xlmr_large_results.json
├── roberta_large_results.json
└── model_comparison.csv
```

The JSON files contain evaluation metrics, per-class reports and confusion matrices.

## Saved models

The trained RoBERTa-large checkpoint is stored separately on Google Drive:

```text
diploma-mer/
└── outputs/
    └── text_roberta_large_ctx3/
        └── saved_model/
            ├── model.safetensors
            ├── config.json
            ├── tokenizer.json
            └── tokenizer_config.json
```

Model weights are excluded from the GitHub repository due to their size.

The saved checkpoint has not yet been independently re-evaluated to confirm its exact correspondence with the reported final metrics.

## Next steps

1. Compare models with and without dialogue context.
2. Analyze confusion matrices and classification errors.
3. Improve recognition of underrepresented emotion classes.
4. Develop a demo for predicting emotions from custom dialogue.
5. Extend the system toward audio and video emotion recognition.

## Conclusion

We completed the first training and evaluation experiments using XLM-RoBERTa-large and RoBERTa-large.

The results establish an initial benchmark for future improvements in contextual and multimodal human emotion recognition.
