"""Fine-tune a transformer encoder for utterance-level emotion recognition on MELD.

Usage:
    python -m mer.text.train --config configs/text_xlmr_large.yaml
    python -m mer.text.train --config configs/smoke.yaml n_context=0 epochs=1

Any key=value after the config path overrides the YAML value.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from mer.data.meld import EMOTIONS, ID2LABEL, LABEL2ID, add_context, download_meld, load_split

DEFAULTS = {
    "model_name": "FacebookAI/xlm-roberta-large",
    "data_dir": "data/meld",
    "output_dir": "outputs/run",  # results.json + config (small; can live on Drive)
    "checkpoint_dir": "checkpoints/run",  # large; keep on local Colab disk, not Drive
    "n_context": 3,  # previous turns given to the model
    "with_speaker": True,
    "max_length": 256,
    "epochs": 6,
    "lr": 1e-5,
    "batch_size": 8,
    "grad_accum": 2,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "class_weights": True,  # inverse-frequency weights in the loss (MELD is very imbalanced)
    "early_stopping_patience": 2,
    "gradient_checkpointing": False,
    "seed": 42,
    "limit_train": None,  # use only N train examples (for smoke tests)
    "limit_eval": None,
    "save_model": False,  # keep the best checkpoint at the end (large: ~2 GB for xlm-r-large)
}


class MeldTextDataset(Dataset):
    def __init__(self, df, tokenizer, max_length: int):
        self.features = []
        for target, context, label in zip(df["target"], df["context"], df["label"]):
            # Target utterance first, context second; only the context is
            # truncated, from the left, so the nearest turns are kept.
            enc = tokenizer(
                target,
                context if context else None,
                truncation="only_second" if context else True,
                max_length=max_length,
            )
            enc["labels"] = int(label)
            self.features.append(enc)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        weight = self.class_weights.to(outputs.logits.device) if self.class_weights is not None else None
        loss = torch.nn.functional.cross_entropy(outputs.logits.float(), labels, weight=weight)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "weighted_f1": f1_score(labels, preds, average="weighted"),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "accuracy": accuracy_score(labels, preds),
    }


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*", help="key=value pairs")
    args = parser.parse_args()
    cfg = dict(DEFAULTS)
    cfg.update(yaml.safe_load(Path(args.config).read_text()) or {})
    for item in args.overrides:
        key, value = item.split("=", 1)
        if key not in DEFAULTS:
            raise KeyError(f"Unknown config key: {key}")
        cfg[key] = yaml.safe_load(value)
    return cfg


def prepare(df, cfg, tokenizer, limit):
    df = add_context(df, cfg["n_context"], cfg["with_speaker"], sep=f" {tokenizer.sep_token} ")
    if limit:
        df = df.sample(n=min(limit, len(df)), random_state=cfg["seed"])
    return df, MeldTextDataset(df, tokenizer, cfg["max_length"])


def main():
    cfg = parse_config()
    random.seed(cfg["seed"])
    set_seed(cfg["seed"])
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    download_meld(cfg["data_dir"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    tokenizer.truncation_side = "left"

    train_df, train_ds = prepare(load_split(cfg["data_dir"], "train"), cfg, tokenizer, cfg["limit_train"])
    dev_df, dev_ds = prepare(load_split(cfg["data_dir"], "dev"), cfg, tokenizer, cfg["limit_eval"])
    test_df, test_ds = prepare(load_split(cfg["data_dir"], "test"), cfg, tokenizer, cfg["limit_eval"])
    print(f"train={len(train_ds)} dev={len(dev_ds)} test={len(test_ds)}")
    print("Example input:", tokenizer.decode(train_ds[0]["input_ids"]))

    class_weights = None
    if cfg["class_weights"]:
        counts = np.bincount(train_df["label"], minlength=len(EMOTIONS)).astype(float)
        w = counts.sum() / (len(EMOTIONS) * np.maximum(counts, 1))
        class_weights = torch.tensor(w, dtype=torch.float)
        print("Class weights:", dict(zip(EMOTIONS, np.round(w, 2))))

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"], num_labels=len(EMOTIONS), id2label=ID2LABEL, label2id=LABEL2ID
    )

    # warmup in steps (TrainingArguments.warmup_ratio was removed in transformers 5)
    steps_per_epoch = math.ceil(len(train_ds) / (cfg["batch_size"] * cfg["grad_accum"]))
    warmup_steps = int(cfg["warmup_ratio"] * steps_per_epoch * cfg["epochs"])

    use_cuda = torch.cuda.is_available()
    bf16 = use_cuda and torch.cuda.is_bf16_supported()
    args = TrainingArguments(
        output_dir=cfg["checkpoint_dir"],
        num_train_epochs=cfg["epochs"],
        learning_rate=cfg["lr"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"] * 2,
        gradient_accumulation_steps=cfg["grad_accum"],
        weight_decay=cfg["weight_decay"],
        warmup_steps=warmup_steps,
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        save_only_model=True,
        load_best_model_at_end=True,
        metric_for_best_model="weighted_f1",
        greater_is_better=True,
        logging_steps=50,
        bf16=bf16,
        fp16=use_cuda and not bf16,
        gradient_checkpointing=cfg["gradient_checkpointing"],
        report_to="none",
        seed=cfg["seed"],
    )
    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg["early_stopping_patience"])],
        class_weights=class_weights,
    )

    start = time.time()
    trainer.train()
    minutes = (time.time() - start) / 60

    results = {"train_minutes": round(minutes, 1)}
    for split, ds in [("dev", dev_ds), ("test", test_ds)]:
        pred = trainer.predict(ds)
        y_pred = pred.predictions.argmax(-1)
        y_true = pred.label_ids
        results[split] = {k.replace("test_", ""): v for k, v in pred.metrics.items()}
        results[split]["per_class"] = classification_report(
            y_true, y_pred, labels=list(range(len(EMOTIONS))), target_names=EMOTIONS,
            output_dict=True, zero_division=0,
        )
        results[split]["confusion_matrix"] = confusion_matrix(
            y_true, y_pred, labels=list(range(len(EMOTIONS)))
        ).tolist()
        print(f"\n== {split} ==")
        print(classification_report(
            y_true, y_pred, labels=list(range(len(EMOTIONS))), target_names=EMOTIONS,
            digits=4, zero_division=0,
        ))

    (out / "results.json").write_text(json.dumps(results, indent=2, default=float))
    if cfg["save_model"]:
        trainer.save_model(str(out / "best_model"))
    print(f"\nDev weighted F1:  {results['dev']['weighted_f1']:.4f}")
    print(f"Test weighted F1: {results['test']['weighted_f1']:.4f}")
    print(f"Results saved to {out / 'results.json'}")


if __name__ == "__main__":
    main()
