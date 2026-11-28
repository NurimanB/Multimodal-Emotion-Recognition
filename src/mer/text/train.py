"""Fine-tune a transformer encoder for utterance-level emotion recognition on MELD.

Usage:
    python -m mer.text.train --config configs/text_xlmr_large.yaml
    python -m mer.text.train --config configs/smoke.yaml n_context=0 epochs=1

Any key=value after the config path overrides the YAML value.
"""
from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from mer.data.meld import EMOTIONS, ID2LABEL, LABEL2ID, add_context, download_meld, load_split
from mer.text.encoding import encode_example
from mer.evaluation.artifacts import metrics as classification_metrics, runtime_info, save_split, write_json
from transformers.trainer_utils import get_last_checkpoint

DEFAULTS = {
    "model_name": "FacebookAI/xlm-roberta-large",
    "data_dir": "data/meld",
    "output_dir": "outputs/run",  # results.json + config (small; can live on Drive)
    "checkpoint_dir": "checkpoints/run",  # large; keep on local Colab disk, not Drive
    "n_context": 3,  # previous turns given to the model
    "n_future": 0,  # next turns given to the model (offline setting)
    "with_speaker": True,
    "max_length": 256,
    "epochs": 6,
    "lr": 1e-5,
    "batch_size": 8,
    "grad_accum": 2,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "class_weights": True,  # inverse-frequency weights in the loss (MELD is very imbalanced)
    "class_weight_power": 1.0,  # 1.0 = inverse frequency, 0.5 = its square root (milder)
    "focal_gamma": 0.0,  # > 0 switches cross-entropy to focal loss
    "early_stopping_patience": 2,
    "gradient_checkpointing": False,
    "seed": 42,
    "limit_train": None,  # use only N train examples (for smoke tests)
    "limit_eval": None,
    "save_model": False,  # keep the best checkpoint at the end (large: ~2 GB for xlm-r-large)
    "evaluate_test": True,  # disable during dev selection
    "save_only_model": True,
    "resume_from_checkpoint": None,  # "auto" requires full optimizer/RNG checkpoints
    "logging_steps": 50,
    "save_probs": True,  # dev/test probabilities of the best model, for ensembles (small .npy files)
}


class MeldTextDataset(Dataset):
    def __init__(self, df, tokenizer, max_length: int):
        self.features = []
        for target, context, label in zip(df["target"], df["context"], df["label"]):
            # Target utterance first, context second; only the context is
            # truncated, from the left, so the nearest turns are kept.
            enc = encode_example(tokenizer, target, context, max_length)
            enc["labels"] = int(label)
            self.features.append(enc)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, focal_gamma=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.model_accepts_loss_kwargs = False  # mean loss per microbatch; Trainer handles accumulation
        self.focal_gamma = focal_gamma

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits.float()
        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        if self.focal_gamma > 0:
            # focal loss: down-weights utterances the model already gets right
            logp = torch.nn.functional.log_softmax(logits, dim=-1).gather(1, labels[:, None]).squeeze(1)
            loss = -((1 - logp.exp()) ** self.focal_gamma) * logp
            if weight is not None:
                loss = loss * weight[labels]
            loss = loss.mean()
        else:
            loss = torch.nn.functional.cross_entropy(logits, labels, weight=weight)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "weighted_f1": f1_score(labels, preds, average="weighted"),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "accuracy": accuracy_score(labels, preds),
    }


def parse_config(defaults=DEFAULTS):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*", help="key=value pairs")
    args = parser.parse_args()
    cfg = dict(defaults)
    cfg.update(yaml.safe_load(Path(args.config).read_text()) or {})
    for item in args.overrides:
        key, value = item.split("=", 1)
        if key not in defaults:
            raise KeyError(f"Unknown config key: {key}")
        cfg[key] = yaml.safe_load(value)
    return cfg


def prepare(df, cfg, tokenizer, limit):
    df = add_context(df, cfg["n_context"], cfg["with_speaker"], sep=f" {tokenizer.sep_token} ",
                     n_future=cfg["n_future"])
    if limit:
        df = df.sample(n=min(limit, len(df)), random_state=cfg["seed"])
    return df, MeldTextDataset(df, tokenizer, cfg["max_length"])


class StateExportCallback(TrainerCallback):
    """Persist small curve evidence even when large checkpoints are local."""
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)

    def on_log(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            state.save_to_json(str(self.output_dir / "trainer_state.json"))

    def on_save(self, args, state, control, **kwargs):
        self.on_log(args, state, control)


def main():
    cfg = parse_config()
    random.seed(cfg["seed"])
    set_seed(cfg["seed"])
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", cfg)

    download_meld(cfg["data_dir"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    tokenizer.truncation_side = "left"

    train_df, train_ds = prepare(load_split(cfg["data_dir"], "train"), cfg, tokenizer, cfg["limit_train"])
    dev_df, dev_ds = prepare(load_split(cfg["data_dir"], "dev"), cfg, tokenizer, cfg["limit_eval"])
    print(f"train={len(train_ds)} dev={len(dev_ds)}; test evaluation={cfg['evaluate_test']}")
    print("Example input:", tokenizer.decode(train_ds[0]["input_ids"]))

    class_weights = None
    if cfg["class_weights"]:
        counts = np.bincount(train_df["label"], minlength=len(EMOTIONS)).astype(float)
        w = (counts.sum() / (len(EMOTIONS) * np.maximum(counts, 1))) ** cfg["class_weight_power"]
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
        save_total_limit=1 if cfg["save_only_model"] else 2,
        save_only_model=cfg["save_only_model"],
        load_best_model_at_end=True,
        metric_for_best_model="weighted_f1",
        greater_is_better=True,
        logging_steps=cfg["logging_steps"],
        bf16=bf16,
        fp16=use_cuda and not bf16,
        gradient_checkpointing=cfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False} if cfg["gradient_checkpointing"] else None,
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg["early_stopping_patience"]),
                   StateExportCallback(out)],
        class_weights=class_weights,
        focal_gamma=cfg["focal_gamma"],
    )

    (out / "training_args.json").write_text(args.to_json_string(), encoding="utf-8")
    resume = cfg["resume_from_checkpoint"]
    if resume == "auto":
        resume = get_last_checkpoint(cfg["checkpoint_dir"]) if Path(cfg["checkpoint_dir"]).is_dir() else None
    if resume and cfg["save_only_model"]:
        raise ValueError("Optimizer/RNG resume requires save_only_model=false")
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    trainer.train(resume_from_checkpoint=resume)
    minutes = (time.time() - start) / 60

    trainer.state.save_to_json(str(out / "trainer_state.json"))
    results = {"train_minutes": round(minutes, 2), "model_name": cfg["model_name"],
               "best_validation_metric": trainer.state.best_metric,
               "checkpoint_selection": "highest dev weighted F1; early stopping on dev"}
    evaluation = [("dev", dev_df, dev_ds)]
    if cfg["evaluate_test"]:
        test_df, test_ds = prepare(load_split(cfg["data_dir"], "test"), cfg, tokenizer, cfg["limit_eval"])
        evaluation.append(("test", test_df, test_ds))
    for split, frame, dataset in evaluation:
        pred = trainer.predict(dataset)
        probs = torch.softmax(torch.tensor(pred.predictions).float(), -1).numpy()
        score = save_split(out, split, frame, probs) if cfg["save_probs"] else classification_metrics(pred.label_ids, probs)
        score.update({k.replace("test_", ""): v for k, v in pred.metrics.items()})
        results[split] = score
        print(f"{split} weighted F1: {score['weighted_f1']:.4f}")
    results["hardware"] = runtime_info(trainer.model)
    results["hardware"].update(training_batch_size=cfg["batch_size"],
                               gradient_accumulation_steps=cfg["grad_accum"], max_length=cfg["max_length"])
    write_json(out / "results.json", results)
    if cfg["save_model"]:
        trainer.save_model(str(out / "best_model"))
        tokenizer.save_pretrained(str(out / "best_model"))
    write_json(out / "checkpoint_manifest.json", {
        "model_name": cfg["model_name"], "best_validation_metric": trainer.state.best_metric,
        "best_training_checkpoint": trainer.state.best_model_checkpoint,
        "exported_checkpoint": str(out / "best_model") if cfg["save_model"] else None,
        "label_order": EMOTIONS, "seed": cfg["seed"],
    })
    print(f"\nDev weighted F1:  {results['dev']['weighted_f1']:.4f}")
    if "test" in results:
        print(f"Test weighted F1: {results['test']['weighted_f1']:.4f}")
    print(f"Results saved to {out / 'results.json'}")


if __name__ == "__main__":
    main()
