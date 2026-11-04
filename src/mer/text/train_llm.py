"""QLoRA fine-tuning of a decoder LLM (default Qwen2.5-7B-Instruct) as a 7-class emotion classifier on MELD.

The LLM is loaded in 4 bit, LoRA adapters are trained on all attention and MLP projections, and a
linear classification head reads the last token. The prompt puts the dialogue context first and the
target utterance last, so truncation (from the left) never cuts the target.

Extra packages (not in requirements.txt):
    pip install -q peft bitsandbytes

Usage:
    python -m mer.text.train_llm --config configs/smoke_qwen.yaml
    python -m mer.text.train_llm --config configs/text_qwen7b_qlora.yaml data_dir=... output_dir=... checkpoint_dir=...

Put checkpoint_dir on Google Drive: LoRA checkpoints are small, and a rerun of the same command after
a Colab disconnect resumes from the last one. Outputs (results.json, dev/test probs and labels) have
the same format as mer.text.train, so scripts/ensemble_eval.py can mix both kinds of runs.
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from mer.data.meld import EMOTIONS, ID2LABEL, LABEL2ID, add_context, download_meld, load_split
from mer.text.train import WeightedTrainer, compute_metrics, parse_config

DEFAULTS = {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "data_dir": "data/meld",
    "output_dir": "outputs/run",
    "checkpoint_dir": "checkpoints/run",  # LoRA checkpoints (~100-200 MB each); put on Drive to resume
    "n_context": 3,
    "n_future": 0,
    "with_speaker": True,
    "max_length": 256,
    "epochs": 2,
    "lr": 1.0e-4,
    "batch_size": 4,
    "grad_accum": 4,
    "weight_decay": 0.0,
    "warmup_ratio": 0.05,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "class_weights": True,
    "class_weight_power": 0.5,
    "focal_gamma": 0.0,
    "eval_steps": 300,  # dev evaluation and checkpoint every N optimizer steps
    "early_stopping_patience": 3,  # in evaluations, not epochs
    "seed": 42,
    "limit_train": None,
    "limit_eval": None,
    "save_probs": True,
}

PROMPT = (
    "Dialogue from the TV series Friends.\n"
    "Previous turns:\n{past}\n"
    "{future_block}"
    "What emotion does the speaker express in this utterance "
    "(neutral, joy, surprise, anger, sadness, disgust, fear)?\n"
    "Utterance: {target}"
)


def build_prompt(target: str, context: str) -> str:
    """Turn the add_context output (past ## future, lines joined by newlines) into the LLM prompt."""
    past, _, future = context.partition(" ## ") if " ## " in context else (context, "", "")
    if context.startswith("## "):  # no past turns, only future ones
        past, future = "", context[3:]
    future_block = f"Next turns:\n{future.strip()}\n" if future.strip() else ""
    return PROMPT.format(past=past.strip() or "(none)", future_block=future_block, target=target)


class PromptDataset(Dataset):
    def __init__(self, df, tokenizer, max_length: int):
        self.features = []
        for target, context, label in zip(df["target"], df["context"], df["label"]):
            enc = tokenizer(build_prompt(target, context), truncation=True, max_length=max_length)
            enc["labels"] = int(label)
            self.features.append(enc)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


def prepare(df, cfg, tokenizer, limit):
    df = add_context(df, cfg["n_context"], cfg["with_speaker"], sep="\n", n_future=cfg["n_future"])
    if limit:
        df = df.sample(n=min(limit, len(df)), random_state=cfg["seed"])
    return df, PromptDataset(df, tokenizer, cfg["max_length"])


def main():
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    cfg = parse_config(DEFAULTS)
    random.seed(cfg["seed"])
    set_seed(cfg["seed"])
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    download_meld(cfg["data_dir"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    tokenizer.truncation_side = "left"  # the target utterance is at the end of the prompt
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_df, train_ds = prepare(load_split(cfg["data_dir"], "train"), cfg, tokenizer, cfg["limit_train"])
    dev_df, dev_ds = prepare(load_split(cfg["data_dir"], "dev"), cfg, tokenizer, cfg["limit_eval"])
    test_df, test_ds = prepare(load_split(cfg["data_dir"], "test"), cfg, tokenizer, cfg["limit_eval"])
    print(f"train={len(train_ds)} dev={len(dev_ds)} test={len(test_ds)}")
    print("Example input:\n" + tokenizer.decode(train_ds[0]["input_ids"]))

    class_weights = None
    if cfg["class_weights"]:
        counts = np.bincount(train_df["label"], minlength=len(EMOTIONS)).astype(float)
        w = (counts.sum() / (len(EMOTIONS) * np.maximum(counts, 1))) ** cfg["class_weight_power"]
        class_weights = torch.tensor(w, dtype=torch.float)
        print("Class weights:", dict(zip(EMOTIONS, np.round(w, 2))))

    bf16 = torch.cuda.is_bf16_supported()  # T4: False -> fp16
    compute_dtype = torch.bfloat16 if bf16 else torch.float16
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"],
        num_labels=len(EMOTIONS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        ),
        torch_dtype=compute_dtype,
        device_map={"": 0},
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        task_type="SEQ_CLS",
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))  # SEQ_CLS keeps the new "score" head trainable and saves it with the adapters
    model.print_trainable_parameters()

    steps_per_epoch = math.ceil(len(train_ds) / (cfg["batch_size"] * cfg["grad_accum"]))
    warmup_steps = int(cfg["warmup_ratio"] * steps_per_epoch * cfg["epochs"])
    args = TrainingArguments(
        output_dir=cfg["checkpoint_dir"],
        num_train_epochs=cfg["epochs"],
        learning_rate=cfg["lr"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"] * 2,
        gradient_accumulation_steps=cfg["grad_accum"],
        weight_decay=cfg["weight_decay"],
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        eval_strategy="steps",
        eval_steps=cfg["eval_steps"],
        save_strategy="steps",
        save_steps=cfg["eval_steps"],
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="weighted_f1",
        greater_is_better=True,
        logging_steps=25,
        bf16=bf16,
        fp16=not bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
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
        focal_gamma=cfg["focal_gamma"],
    )

    last = get_last_checkpoint(cfg["checkpoint_dir"]) if Path(cfg["checkpoint_dir"]).is_dir() else None
    if last:
        print(f"Resuming from {last}")
    start = time.time()
    trainer.train(resume_from_checkpoint=last)
    minutes = (time.time() - start) / 60

    results = {"train_minutes": round(minutes, 1), "resumed_from": last}
    for split, ds in [("dev", dev_ds), ("test", test_ds)]:
        pred = trainer.predict(ds)
        y_pred = pred.predictions.argmax(-1)
        y_true = pred.label_ids
        if cfg["save_probs"]:
            np.save(out / f"{split}_probs.npy", torch.softmax(torch.tensor(pred.predictions).float(), -1).numpy())
            np.save(out / f"{split}_labels.npy", y_true)
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
    trainer.save_model(str(out / "best_adapter"))  # LoRA adapter + score head only, ~100-200 MB
    print(f"\nDev weighted F1:  {results['dev']['weighted_f1']:.4f}")
    print(f"Test weighted F1: {results['test']['weighted_f1']:.4f}")
    print(f"Results saved to {out / 'results.json'}")


if __name__ == "__main__":
    main()
