"""Freeze blend weights/bias using dev only; evaluate the frozen blend on request."""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score

from mer.data.meld import load_split
from mer.evaluation.artifacts import load_aligned, save_split, validate_probs, write_json


def blend(probabilities, weights, neutral_bias=0.0):
    if len(probabilities) != len(weights) or min(weights) < 0 or not np.isclose(sum(weights), 1):
        raise ValueError("Weights must be nonnegative, sum to one, and match the number of runs")
    result = sum(w * validate_probs(p) for w, p in zip(weights, probabilities))
    logp = np.log(np.clip(result, 1e-12, 1))
    logp[:, 0] += neutral_bias
    exp = np.exp(logp - logp.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def select_blend(labels, probabilities):
    n = len(probabilities)
    if not 1 <= n <= 4:
        raise ValueError("Select 1-4 runs; average seeds within each architecture before larger searches")
    weights = [tuple(value / 4 for value in row) for row in itertools.product(range(5), repeat=n) if sum(row) == 4]
    equal = tuple([1 / n] * n)
    weights = [equal] + [row for row in weights if row != equal]
    best, best_score = None, -1.0
    for row in weights:
        for bias in [0.0, -0.5, -0.25, 0.25, 0.5]:
            score = float(f1_score(labels, blend(probabilities, row, bias).argmax(axis=1), average="weighted"))
            if score > best_score:
                best, best_score = {"weights": list(row), "neutral_bias": bias}, score
    return {**best, "dev_weighted_f1": best_score}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*")
    parser.add_argument("--data-dir", default="data/meld")
    parser.add_argument("--output-dir", default="outputs/experiments/ensemble")
    parser.add_argument("--allow-legacy-order", action="store_true")
    parser.add_argument("--evaluate-frozen", action="store_true")
    args = parser.parse_args()
    out = Path(args.output_dir)
    selection_file = out / "selection.json"
    dev = load_split(args.data_dir, "dev")
    if args.evaluate_frozen:
        selection = json.loads(selection_file.read_text(encoding="utf-8"))
        if args.runs and [str(Path(run).resolve()) for run in args.runs] != selection["runs"]:
            raise ValueError("Runs differ from the frozen dev selection")
    else:
        if not args.runs:
            parser.error("Provide run directories for dev selection")
        dev_probs = [load_aligned(run, "dev", dev, args.allow_legacy_order) for run in args.runs]
        selection = select_blend(dev.label.to_numpy(), dev_probs)
        selection.update(runs=[str(Path(run).resolve()) for run in args.runs],
                         selection_split="dev", label_order="mer.data.meld.EMOTIONS",
                         allow_legacy_order=args.allow_legacy_order)
        write_json(selection_file, selection)
        write_json(out / "results.json", {
            "model_name": "Dev-selected probability ensemble",
            "dev": save_split(out, "dev", dev, blend(dev_probs, selection["weights"], selection["neutral_bias"])),
        })
        print("Frozen dev selection:", selection)
        return
    test = load_split(args.data_dir, "test")
    runs, allow = selection["runs"], bool(selection["allow_legacy_order"])
    result = {"model_name": "Dev-selected probability ensemble", "selection": selection}
    for split, frame in [("dev", dev), ("test", test)]:
        probs = [load_aligned(run, split, frame, allow) for run in runs]
        result[split] = save_split(out, split, frame, blend(probs, selection["weights"], selection["neutral_bias"]))
    write_json(out / "results.json", result)
    print("Frozen ensemble test weighted F1:", result["test"]["weighted_f1"])


if __name__ == "__main__":
    main()
