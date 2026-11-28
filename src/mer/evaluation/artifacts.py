"""Shared label order, sample identities, metrics and measured runtime metadata."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss

from mer.data.meld import EMOTIONS


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def sample_ids(df):
    ids = [f"{int(d)}:{int(u)}" for d, u in zip(df.dialogue_id, df.utterance_id)]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate dialogue/utterance keys")
    return ids


def fingerprint(df):
    rows = df[["dialogue_id", "utterance_id", "speaker", "utterance", "label"]].to_dict("records")
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_probs(probs, n_rows=None):
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] != len(EMOTIONS):
        raise ValueError("Expected an N x 7 probability matrix in MELD label order")
    if n_rows is not None and len(probs) != n_rows:
        raise ValueError("Prediction count does not match the evaluation split")
    if not np.isfinite(probs).all() or (probs < 0).any():
        raise ValueError("Probabilities must be finite and nonnegative")
    if not np.allclose(probs.sum(axis=1), 1, atol=1e-4):
        raise ValueError("Probability rows must sum to one")
    return probs


def metrics(y, probs):
    probs = validate_probs(probs, len(y))
    predicted = probs.argmax(axis=1)
    report = classification_report(
        y, predicted, labels=list(range(7)), target_names=EMOTIONS,
        output_dict=True, zero_division=0,
    )
    return {
        "n_examples": len(y),
        "accuracy": float(accuracy_score(y, predicted)),
        "weighted_f1": report["weighted avg"]["f1-score"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_precision": report["weighted avg"]["precision"],
        "weighted_recall": report["weighted avg"]["recall"],
        "cross_entropy": float(log_loss(y, probs, labels=list(range(7)))),
        "per_class": report,
        "confusion_matrix": confusion_matrix(y, predicted, labels=list(range(7))).tolist(),
    }


def save_split(out, split, df, probs):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    probs = validate_probs(probs, len(df))
    y = df.label.to_numpy(dtype=np.int64)
    ids = sample_ids(df)
    np.save(out / f"{split}_probs.npy", probs.astype(np.float32))
    np.save(out / f"{split}_labels.npy", y)
    write_json(out / f"{split}_ids.json", ids)
    write_json(out / f"{split}_manifest.json", {
        "split": split, "label_order": EMOTIONS, "n_examples": len(df),
        "dataset_fingerprint": fingerprint(df),
    })
    predicted = probs.argmax(axis=1)
    examples = df[["dialogue_id", "utterance_id", "speaker", "utterance"]].copy()
    examples.insert(0, "sample_id", ids)
    examples["context"] = df["context"].to_numpy() if "context" in df else ""
    examples["gold"] = [EMOTIONS[int(i)] for i in y]
    examples["predicted"] = [EMOTIONS[int(i)] for i in predicted]
    examples["confidence"] = probs.max(axis=1)
    examples["correct"] = y == predicted
    examples.to_csv(out / f"{split}_predictions.csv", index=False)
    return metrics(y, probs)


def load_aligned(run, split, df, allow_legacy=False):
    run = Path(run)
    probs = validate_probs(np.load(run / f"{split}_probs.npy", allow_pickle=False), len(df))
    labels = np.load(run / f"{split}_labels.npy", allow_pickle=False)
    if not np.array_equal(labels, df.label.to_numpy()):
        raise ValueError(f"{run}: {split} labels/order differ from full canonical MELD")
    ids_path = run / f"{split}_ids.json"
    manifest_path = run / f"{split}_manifest.json"
    if ids_path.exists() and manifest_path.exists():
        if json.loads(ids_path.read_text(encoding="utf-8")) != sample_ids(df):
            raise ValueError(f"{run}: sample identities differ")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["label_order"] != EMOTIONS or manifest["dataset_fingerprint"] != fingerprint(df):
            raise ValueError(f"{run}: dataset or label order differs")
    elif not allow_legacy:
        raise ValueError(f"{run}: missing identities; use --allow-legacy-order only for the old sorted MELD pipeline")
    return probs


def runtime_info(model=None):
    versions = {}
    for name in ["numpy", "pandas", "scikit-learn", "torch", "transformers", "accelerate"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    result = {"python": platform.python_version(), "platform": platform.platform(),
              "processor": platform.processor(), "versions": versions, "device": "cpu"}
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        result["source_commit"] = revision.stdout.strip() if revision.returncode == 0 else None
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5)
        result["source_has_uncommitted_changes"] = bool(dirty.stdout.strip()) if dirty.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        result["source_commit"] = None
    if model is not None:
        import torch
        result.update(parameters=sum(p.numel() for p in model.parameters()),
                      trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
        device = next(model.parameters()).device
        result["device"] = str(device)
        if device.type == "cuda":
            properties = torch.cuda.get_device_properties(device)
            result.update(gpu=properties.name, gpu_total_gib=properties.total_memory / 2**30,
                          peak_cuda_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30)
    return result


def benchmark(predict, warmup=3, repeats=30, synchronize=lambda: None):
    if repeats < 1:
        raise ValueError("Benchmark repeats must be positive")
    for _ in range(warmup):
        predict()
    times = []
    for _ in range(repeats):
        synchronize()
        start = time.perf_counter()
        predict()
        synchronize()
        times.append((time.perf_counter() - start) * 1000)
    return {"batch_size": 1, "warmup": warmup, "repeats": repeats,
            "scope": "end-to-end preprocessing + forward + postprocessing; excludes loading",
            "mean_ms": float(np.mean(times)), "p50_ms": float(np.median(times)),
            "p95_ms": float(np.percentile(times, 95)),
            "throughput_examples_per_second": float(1000 / np.mean(times))}
