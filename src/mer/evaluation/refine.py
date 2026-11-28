"""Refine a frozen two-model ensemble using dev only; evaluate once after freezing.

Works directly from the lightweight Colab evidence ZIP, with no model weights.
This is explicitly an exploratory iteration after the original test was inspected.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np
import pandas as pd

from mer.data.meld import EMOTIONS, LABEL2ID, load_split
from mer.evaluation.artifacts import fingerprint, metrics, sample_ids, save_split, validate_probs, write_json
from mer.evaluation.ensemble import blend

WEIGHTS = [i / 20 for i in range(21)]
BIASES = [i / 20 for i in range(-10, 21)]
GRID = {
    "first_model_weights": WEIGHTS,
    "neutral_biases": BIASES,
    "tie_break": "nearest reference weights/bias; then smaller absolute bias; then smaller first weight",
}
BOOTSTRAP = {"unit": "dialogue", "paired": True, "resamples": 2000, "seed": 2025,
             "interval": "95% percentile", "metric": "weighted_f1 difference: refined minus reference"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Evidence:
    def __init__(self, path):
        self.path = Path(path)
        self.hash = sha256(path)
        self.archive = zipfile.ZipFile(path)
        entries = self.archive.infolist()
        names = [entry.filename for entry in entries]
        if len(names) != len(set(names)) or sum(entry.file_size for entry in entries) > 64 * 1024**2:
            raise ValueError("Duplicate members or unexpectedly large evidence archive")
        for name in names:
            parsed = PurePosixPath(name)
            if parsed.is_absolute() or ".." in parsed.parts or "\\" in name or ":" in name:
                raise ValueError("Unsafe archive member")
        self.original = self.read_json("ensemble/selection.json")
        self.runs = [PurePosixPath(run).name for run in self.original["runs"]]
        if len(self.runs) != 2 or len(set(self.runs)) != 2:
            raise ValueError("This refinement is defined for the original two distinct model runs")
        self.verify_original_freeze()

    def close(self):
        self.archive.close()

    def read_json(self, name):
        return json.loads(self.archive.read(name))

    def verify_original_freeze(self):
        protocol = self.read_json("frozen_protocol.json")
        root = PurePosixPath(protocol["best_new"]).parent
        for path, expected in protocol["artifact_hashes"].items():
            relative = PurePosixPath(path).relative_to(root).as_posix()
            if hashlib.sha256(self.archive.read(relative)).hexdigest() != expected:
                raise ValueError(f"Original frozen artifact changed: {relative}")
        if hashlib.sha256(self.archive.read("ensemble/selection.json")).hexdigest() != protocol["ensemble_selection_sha256"]:
            raise ValueError("Original ensemble selection changed after freeze")

    def load(self, split, data_dir=None):
        first = self.runs[0]
        frame = pd.read_csv(io.BytesIO(self.archive.read(f"{first}/{split}_predictions.csv")),
                            keep_default_na=False)
        required = {"dialogue_id", "utterance_id", "speaker", "utterance", "gold"}
        if not required.issubset(frame.columns) or frame.empty:
            raise ValueError("Missing evaluation example metadata")
        if not set(frame.gold).issubset(LABEL2ID):
            raise ValueError("Unknown gold label")
        frame["label"] = frame.gold.map(LABEL2ID).astype(np.int64)
        frame["emotion"] = frame.gold
        ids = sample_ids(frame)
        digest = fingerprint(frame)
        arrays = []
        for run in self.runs:
            labels = np.load(io.BytesIO(self.archive.read(f"{run}/{split}_labels.npy")), allow_pickle=False)
            probs = np.load(io.BytesIO(self.archive.read(f"{run}/{split}_probs.npy")), allow_pickle=False)
            manifest = self.read_json(f"{run}/{split}_manifest.json")
            if not np.array_equal(labels, frame.label.to_numpy()):
                raise ValueError(f"{run}: labels/order mismatch")
            if self.read_json(f"{run}/{split}_ids.json") != ids:
                raise ValueError(f"{run}: sample identities mismatch")
            if manifest.get("label_order") != EMOTIONS or manifest.get("dataset_fingerprint") != digest:
                raise ValueError(f"{run}: label order/dataset fingerprint mismatch")
            if manifest.get("n_examples") != len(frame) or manifest.get("split") != split:
                raise ValueError(f"{run}: split metadata mismatch")
            arrays.append(validate_probs(probs, len(frame)))
        if data_dir is not None and fingerprint(load_split(data_dir, split)) != digest:
            raise ValueError("Archive does not match the canonical MELD CSV")
        return frame, arrays


def f1_from_matrix(matrix):
    matrix = np.asarray(matrix)
    support = matrix.sum(axis=-1)
    denominator = support + matrix.sum(axis=-2)
    tp = np.diagonal(matrix, axis1=-2, axis2=-1)
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator != 0)
    return (f1 * support).sum(axis=-1) / support.sum(axis=-1)


def fast_f1(labels, predicted):
    matrix = np.bincount(7 * labels + predicted, minlength=49).reshape(7, 7)
    return float(f1_from_matrix(matrix))


def read_frozen(output_dir):
    out = Path(output_dir)
    path = out / "selection.json"
    if (out / "selection.sha256").read_text(encoding="utf-8").strip() != sha256(path):
        raise ValueError("Refinement selection was modified after freezing")
    return json.loads(path.read_text(encoding="utf-8"))


def select(archive, output_dir, data_dir=None):
    out = Path(output_dir)
    evidence = Evidence(archive)
    try:
        if (out / "selection.json").exists():
            previous = read_frozen(out)
            if previous["source_archive_sha256"] != evidence.hash or previous["grid"] != GRID:
                raise ValueError("Output belongs to another experiment; choose a new output directory")
            return previous
        frame, probabilities = evidence.load("dev", data_dir)
        labels = frame.label.to_numpy()
        reference = evidence.original
        reference_prediction = blend(probabilities, reference["weights"], reference["neutral_bias"]).argmax(1)
        reference_f1 = fast_f1(labels, reference_prediction)
        if not np.isclose(reference_f1, reference["dev_weighted_f1"], atol=1e-12):
            raise ValueError("Cannot reproduce the original dev ensemble")
        rows = []
        best_key, best = None, None
        for weight in WEIGHTS:
            logits = np.log(np.clip(weight * probabilities[0] + (1 - weight) * probabilities[1], 1e-12, 1))
            for bias in BIASES:
                adjusted = logits.copy()
                adjusted[:, 0] += bias
                prediction = adjusted.argmax(1)
                score = fast_f1(labels, prediction)
                distance = (weight - reference["weights"][0])**2 + (bias - reference["neutral_bias"])**2
                key = (score, -distance, -abs(bias), -weight)
                row = {"first_weight": weight, "second_weight": 1 - weight,
                       "neutral_bias": bias, "dev_weighted_f1": score,
                       "changed_predictions_vs_v1": int((prediction != reference_prediction).sum())}
                rows.append(row)
                if best_key is None or key > best_key:
                    best_key, best = key, row
        frozen = {
            "schema_version": 1, "source_archive_sha256": evidence.hash,
            "runs": evidence.runs, "weights": [best["first_weight"], best["second_weight"]],
            "neutral_bias": best["neutral_bias"], "dev_weighted_f1": best["dev_weighted_f1"],
            "reference_weights": reference["weights"], "reference_neutral_bias": reference["neutral_bias"],
            "reference_dev_weighted_f1": reference_f1,
            "dev_dataset_fingerprint": fingerprint(frame), "label_order": EMOTIONS,
            "selection_split": "dev", "test_used_for_selection": False,
            "original_test_previously_inspected": True,
            "interpretation": "Exploratory refinement after the v1 test report, not a newly unseen benchmark",
            "grid": GRID, "candidate_count": len(rows), "bootstrap_protocol": BOOTSTRAP,
            "canonical_csv_checked": data_dir is not None,
        }
        write_json(out / "selection.json", frozen)
        (out / "selection.sha256").write_text(sha256(out / "selection.json") + "\n", encoding="utf-8")
        pd.DataFrame(rows).to_csv(out / "dev_search.csv", index=False)
        write_json(out / "results.json", {
            "model_name": "Dev-refined RoBERTa + DeBERTa (v2)",
            "dev": save_split(out, "dev", frame, blend(probabilities, frozen["weights"], frozen["neutral_bias"])),
            "selection": frozen,
        })
        return frozen
    finally:
        evidence.close()


def paired_bootstrap(frame, reference_prediction, refined_prediction):
    labels = frame.label.to_numpy()
    _, groups = np.unique(frame.dialogue_id.to_numpy(), return_inverse=True)
    n_groups = int(groups.max()) + 1
    reference = np.zeros((n_groups, 7, 7), dtype=np.int64)
    refined = np.zeros_like(reference)
    np.add.at(reference, (groups, labels, reference_prediction), 1)
    np.add.at(refined, (groups, labels, refined_prediction), 1)
    rng = np.random.default_rng(BOOTSTRAP["seed"])
    counts = rng.multinomial(n_groups, np.full(n_groups, 1 / n_groups), size=BOOTSTRAP["resamples"])
    differences = f1_from_matrix(np.tensordot(counts, refined, axes=(1, 0))) - f1_from_matrix(
        np.tensordot(counts, reference, axes=(1, 0)))
    return {**BOOTSTRAP, "n_dialogues": n_groups,
            "lower": float(np.percentile(differences, 2.5)),
            "upper": float(np.percentile(differences, 97.5)),
            "note": "Within-benchmark uncertainty only; does not undo prior test inspection"}


def evaluate(archive, output_dir, data_dir=None):
    out = Path(output_dir)
    frozen = read_frozen(out)
    evidence = Evidence(archive)
    try:
        if frozen["source_archive_sha256"] != evidence.hash or frozen["runs"] != evidence.runs:
            raise ValueError("Evaluation archive differs from the frozen dev experiment")
        if (out / "evaluation.json").exists():
            return json.loads((out / "evaluation.json").read_text(encoding="utf-8"))
        dev, dev_probs = evidence.load("dev", data_dir)
        actual = metrics(dev.label.to_numpy(), blend(dev_probs, frozen["weights"], frozen["neutral_bias"]))
        if not np.isclose(actual["weighted_f1"], frozen["dev_weighted_f1"], atol=1e-12):
            raise ValueError("Frozen dev result no longer matches")
        frame, probabilities = evidence.load("test", data_dir)
        original = blend(probabilities, frozen["reference_weights"], frozen["reference_neutral_bias"])
        refined = blend(probabilities, frozen["weights"], frozen["neutral_bias"])
        reference_metrics = metrics(frame.label.to_numpy(), original)
        refined_metrics = save_split(out, "test", frame, refined)
        difference = refined_metrics["weighted_f1"] - reference_metrics["weighted_f1"]
        interval = paired_bootstrap(frame, original.argmax(1), refined.argmax(1))
        evaluation = {
            "source_archive_sha256": evidence.hash, "selection_sha256": sha256(out / "selection.json"),
            "reference_test": reference_metrics, "refined_test": refined_metrics,
            "weighted_f1_difference": difference, "paired_dialogue_bootstrap": interval,
            "test_used_for_selection": False, "original_test_previously_inspected": True,
            "no_training_required": True,
        }
        write_json(out / "evaluation.json", evaluation)
        result = json.loads((out / "results.json").read_text(encoding="utf-8"))
        result["test"] = refined_metrics
        result["evaluation"] = evaluation
        write_json(out / "results.json", result)
        pd.DataFrame([
            {"version": "original_v1", "dev_weighted_f1": frozen["reference_dev_weighted_f1"],
             "test_weighted_f1": reference_metrics["weighted_f1"], "test_macro_f1": reference_metrics["macro_f1"]},
            {"version": "dev_refinement_v2", "dev_weighted_f1": frozen["dev_weighted_f1"],
             "test_weighted_f1": refined_metrics["weighted_f1"], "test_macro_f1": refined_metrics["macro_f1"]},
        ]).to_csv(out / "comparison.csv", index=False)
        return evaluation
    finally:
        evidence.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["select", "evaluate"])
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output-dir", default="outputs/ensemble_refinement_v2")
    parser.add_argument("--data-dir", help="Optional canonical MELD CSV directory for independent fingerprint checks")
    args = parser.parse_args()
    function = select if args.action == "select" else evaluate
    print(json.dumps(function(args.archive, args.output_dir, args.data_dir), indent=2))


if __name__ == "__main__":
    main()
