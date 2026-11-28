"""Export measured confusion matrices, class reports, learning curves and examples."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mer.data.meld import EMOTIONS
from mer.evaluation.artifacts import write_json


def plot_curves(state_path, out):
    history = pd.DataFrame(json.loads(Path(state_path).read_text(encoding="utf-8")).get("log_history", []))
    if history.empty:
        return False
    history.to_csv(out / "training_history.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    found = False
    for key, axis, label in [
        ("loss", axes[0], "Training batch loss"), ("eval_loss", axes[0], "Validation loss"),
        ("eval_weighted_f1", axes[1], "Validation weighted F1"), ("eval_macro_f1", axes[1], "Validation macro F1"),
    ]:
        if key in history:
            rows = history.dropna(subset=[key])
            if not rows.empty:
                x = rows["epoch"] if "epoch" in rows else rows["step"]
                axis.plot(x, rows[key], marker="o", markersize=3, label=label)
                found = True
    axes[0].set_ylabel("Loss")
    axes[1].set_ylabel("F1")
    for axis in axes:
        axis.set_xlabel("Epoch" if "epoch" in history else "Optimizer step")
        axis.grid(alpha=0.2)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=8)
    fig.tight_layout()
    if found:
        fig.savefig(out / "learning_curves.png", dpi=160)
    plt.close(fig)
    return found


def plot_confusion(matrix, path, title, normalized=False):
    matrix = np.asarray(matrix)
    if matrix.shape != (7, 7) or (matrix < 0).any():
        raise ValueError("Invalid confusion matrix")
    shown = matrix.astype(float)
    if normalized:
        shown = np.divide(shown, shown.sum(axis=1, keepdims=True),
                          out=np.zeros_like(shown), where=shown.sum(axis=1, keepdims=True) != 0)
    fig, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(shown, cmap="Blues", vmin=0, vmax=1 if normalized else None)
    axis.set(xticks=range(7), yticks=range(7), xticklabels=EMOTIONS, yticklabels=EMOTIONS,
             xlabel="Predicted emotion", ylabel="True emotion", title=title)
    plt.setp(axis.get_xticklabels(), rotation=40, ha="right")
    for i in range(7):
        for j in range(7):
            label = f"{shown[i, j]:.2f}" if normalized else str(int(matrix[i, j]))
            axis.text(j, i, label, ha="center", va="center",
                      color="white" if shown[i, j] > float(shown.max()) / 2 else "black", fontsize=8)
    fig.colorbar(image, ax=axis, label="Recall fraction" if normalized else "Utterances")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def select_examples(frame, per_kind=5):
    chosen = []
    for correct in [True, False]:
        pool = frame.loc[frame.correct.eq(correct)].sort_values("confidence", ascending=False)
        diverse = pool.drop_duplicates("gold").head(per_kind)
        chosen.append(pd.concat([diverse, pool.drop(index=diverse.index)]).head(per_kind))
    result = pd.concat(chosen, ignore_index=True)
    result["human_error_hypothesis"] = ""
    result["human_notes"] = ""
    return result


def make_report(run_dir, output_dir=None, results_file=None, trainer_state=None, title=None):
    run = Path(run_dir)
    out = Path(output_dir) if output_dir else run / "report"
    out.mkdir(parents=True, exist_ok=True)
    result_path = Path(results_file) if results_file else run / "results.json"
    results = json.loads(result_path.read_text(encoding="utf-8"))
    title = title or results.get("model_name") or run.name
    notes = []
    lines = [f"# {title}", "", "All values below come from recorded artifacts; missing evidence is stated explicitly.", ""]
    for split in ["dev", "test"]:
        if split not in results:
            notes.append(f"{split}: not evaluated in this artifact.")
            continue
        score = results[split]
        lines.extend([f"## {split.capitalize()} split", "",
                      f"- Weighted F1: {score.get('weighted_f1', 'not measured')}",
                      f"- Macro F1: {score.get('macro_f1', 'not measured')}",
                      f"- Accuracy: {score.get('accuracy', 'not measured')}", ""])
        if "confusion_matrix" in score:
            matrix = score["confusion_matrix"]
            plot_confusion(matrix, out / f"{split}_confusion_matrix.png", f"{title}: {split}")
            plot_confusion(matrix, out / f"{split}_confusion_normalized.png", f"{title}: {split} (row-normalized)", True)
            pd.DataFrame(matrix, index=EMOTIONS, columns=EMOTIONS).to_csv(out / f"{split}_confusion_matrix.csv")
            lines.extend([f"![Confusion matrix]({split}_confusion_matrix.png)", ""])
        if "per_class" in score:
            table = pd.DataFrame([{"emotion": name, **score["per_class"][name]} for name in EMOTIONS])
            table = table.rename(columns={"f1-score": "f1"})
            table.to_csv(out / f"{split}_per_class.csv", index=False)
            lines.extend(["| Emotion | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|"])
            for row in table.itertuples(index=False):
                lines.append(f"| {row.emotion} | {row.precision:.4f} | {row.recall:.4f} | {row.f1:.4f} | {int(row.support)} |")
            weakest = table.sort_values("f1").head(2).emotion.tolist()
            lines.extend(["", f"Lowest class F1: {', '.join(weakest)}. Inspect real errors before assigning causes.", ""])
        predictions = run / f"{split}_predictions.csv"
        if predictions.exists():
            select_examples(pd.read_csv(predictions)).to_csv(out / f"{split}_review_examples.csv", index=False)
            lines.extend([f"Examples: [{split}_review_examples.csv]({split}_review_examples.csv).",
                          "Selection: up to five correct and five incorrect; class diversity then confidence.",
                          "Diagnostic examples do not estimate the frequency of errors.", ""])
        else:
            notes.append(f"{split}: per-example predictions unavailable; evaluate a saved checkpoint for real examples.")
    state = Path(trainer_state) if trainer_state else run / "trainer_state.json"
    if state.exists() and plot_curves(state, out):
        lines.extend(["## Learning curves", "", "![Learning curves](learning_curves.png)", "",
                      "Training points are logged batch losses; validation points use the validation set.", ""])
    else:
        notes.append("Learning curves: trainer_state.json/log_history unavailable; no curve was invented.")
    if results.get("hardware"):
        write_json(out / "hardware.json", {"evaluation_or_current": results["hardware"],
                                           "training": results.get("training_hardware")})
        lines.extend(["## Measured hardware and efficiency", "", "See [hardware.json](hardware.json).", "",
                      "Latency is batch 1 including preprocessing, forward and postprocessing, excluding loading.", ""])
    else:
        notes.append("Hardware, parameter count and latency: not recorded in this result artifact.")
    config = run / "config.json"
    if config.exists():
        lines.extend(["## Run configuration", "", "~~~json", config.read_text(encoding="utf-8"), "~~~", ""])
    lines.extend(["## Evidence gaps / limitations", ""] + [f"- {note}" for note in notes])
    lines.extend(["", "Text-only English MELD / Friends dialogues; multilingual and clinical performance is unvalidated.",
                  "Softmax confidence is not a calibrated probability of correctness.",
                  "Test error inspection is final reporting, not another hyperparameter-selection round.", ""])
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(out / "evidence_status.json", {"source": str(result_path), "gaps": notes,
                                            "trainer_state_source": str(state) if state.exists() else None,
                                            "checkpoint": results.get("checkpoint")})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--results-file")
    parser.add_argument("--trainer-state")
    parser.add_argument("--title")
    print(make_report(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
