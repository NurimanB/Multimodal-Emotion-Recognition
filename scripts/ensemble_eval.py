"""Average the saved probabilities of several runs and tune a neutral-class bias on dev.

Each run directory is an output_dir of mer.text.train with dev_probs.npy, dev_labels.npy,
test_probs.npy and test_labels.npy (written when save_probs is on). Runs must use the full
dev/test sets (no limit_eval). The bias is chosen on dev and applied unchanged to test.

Usage:
    python scripts/ensemble_eval.py outputs/text_xlmr_large_ctx3 outputs/text_roberta_large_ctx3
"""
import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

NEUTRAL = 0  # label order in mer.data.meld.EMOTIONS


def scores(labels, probs, bias=0.0):
    logp = np.log(probs + 1e-9)
    logp[:, NEUTRAL] += bias
    preds = logp.argmax(-1)
    return {
        "weighted_f1": round(f1_score(labels, preds, average="weighted"), 4),
        "macro_f1": round(f1_score(labels, preds, average="macro"), 4),
        "accuracy": round(accuracy_score(labels, preds), 4),
    }


def load(runs, split):
    labels = np.load(Path(runs[0]) / f"{split}_labels.npy")
    probs = []
    for run in runs:
        if not np.array_equal(np.load(Path(run) / f"{split}_labels.npy"), labels):
            raise ValueError(f"{run}: {split} examples are in a different order or subset")
        probs.append(np.load(Path(run) / f"{split}_probs.npy"))
    return labels, probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="output_dir of each run")
    args = parser.parse_args()

    dev_y, dev_p = load(args.runs, "dev")
    test_y, test_p = load(args.runs, "test")
    for run, p in zip(args.runs, test_p):
        print(f"{run}  test {scores(test_y, p)}")

    dev_avg, test_avg = np.mean(dev_p, axis=0), np.mean(test_p, axis=0)
    print(f"ensemble  test {scores(test_y, test_avg)}")

    grid = np.round(np.arange(-1.0, 1.01, 0.05), 2)
    best = max(grid, key=lambda b: (scores(dev_y, dev_avg, b)["weighted_f1"], -abs(b)))
    print(f"ensemble + neutral bias {best} (chosen on dev)  dev {scores(dev_y, dev_avg, best)}"
          f"  test {scores(test_y, test_avg, best)}")


if __name__ == "__main__":
    main()
