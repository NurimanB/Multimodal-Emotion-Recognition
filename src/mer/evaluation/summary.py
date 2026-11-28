"""Compare recorded results and report seed variability without filling missing scores."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd

from mer.evaluation.artifacts import write_json


def summarize(runs, output_dir):
    rows = []
    for run in map(Path, runs):
        result = json.loads((run / "results.json").read_text(encoding="utf-8"))
        config_path = run / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        hardware = result.get("hardware", {})
        for split in ["dev", "test"]:
            if split in result:
                metric = result[split]
                rows.append({
                    "run": run.name, "model": config.get("model_name", result.get("model_name", run.name)),
                    "seed": config.get("seed"), "split": split, "n_examples": metric.get("n_examples"),
                    "accuracy": metric.get("accuracy"), "weighted_precision": metric.get("weighted_precision"),
                    "weighted_recall": metric.get("weighted_recall"), "weighted_f1": metric.get("weighted_f1"),
                    "macro_f1": metric.get("macro_f1"), "train_minutes": result.get("train_minutes"),
                    "parameters": hardware.get("parameters"), "device": hardware.get("device"),
                    "latency_p50_ms": hardware.get("latency", {}).get("p50_ms"),
                })
    table = pd.DataFrame(rows)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "comparison.csv", index=False)
    # Distinct seeds only; same run should not appear twice in the input list.
    if table.duplicated(["run", "split"]).any():
        raise ValueError("Duplicate runs in comparison")
    with_seed = table.loc[table.seed.notna()].copy()
    if not with_seed.empty:
        groups = with_seed.groupby(["model", "split"])
        variation = groups.weighted_f1.agg(["count", "mean", "std"]).reset_index()
        variation.to_csv(out / "seed_variation.csv", index=False)
    write_json(out / "reporting_notes.json", {
        "missing_values": "not measured; never replace with guessed values",
        "seed_std": "sample standard deviation (ddof=1); unavailable for a single seed",
        "reported_ensemble_0_664": "Not inserted; a computed ensemble artifact is required",
    })
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--output-dir", default="outputs/experiments/comparison")
    args = parser.parse_args()
    print(summarize(args.runs, args.output_dir).to_string(index=False))


if __name__ == "__main__":
    main()
