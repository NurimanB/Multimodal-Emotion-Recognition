"""Additional DeBERTa seeds 123/2025, preserving v1 and reusing seed 42.

Can run as a CLI script or be pasted as ONE new cell into the existing Colab.
No settings are selected on test. This is a declared follow-up to the v1 test.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import numpy as np
import torch
import yaml

from mer.data.meld import load_split, add_context
from mer.evaluation.artifacts import load_aligned, save_split, write_json
from mer.evaluation.ensemble import blend
from mer.evaluation.report import make_report
from mer.evaluation.summary import summarize
from mer.evaluation.paths import locate_run

MS_SEEDS = (42, 123, 2025)
MS_EXTRA_SEEDS = (123, 2025)
MS_GRID = {"roberta_weights": [i / 20 for i in range(21)],
           "neutral_biases": [i / 20 for i in range(-10, 21)]}
MS_STORAGE_KEYS = {
    "seed", "data_dir", "output_dir", "checkpoint_dir", "resume_from_checkpoint",
    "save_only_model", "save_model", "save_probs", "evaluate_test", "logging_steps",
}


def ms_read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ms_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ms_learning_config(config):
    return {key: value for key, value in config.items() if key not in MS_STORAGE_KEYS}


def ms_stream(module, arguments, log_file):
    command = [sys.executable, "-u", "-m", module, *map(str, arguments)]
    print("\nRUNNING:", " ".join(command), flush=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "TQDM_MININTERVAL": "10"}
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    with Path(log_file).open("a", encoding="utf-8") as log:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace",
                              bufsize=1, env=env) as process:
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
            except KeyboardInterrupt:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
            if code:
                raise subprocess.CalledProcessError(code, command)


def ms_snapshot(source, target, include_test=False):
    source, target = Path(source), Path(target)
    target.mkdir(parents=True, exist_ok=True)
    names = ["config.json", "trainer_state.json", "checkpoint_manifest.json"]
    for split in ["dev"] + (["test"] if include_test else []):
        names += [f"{split}_{suffix}" for suffix in [
            "probs.npy", "labels.npy", "ids.json", "manifest.json", "predictions.csv",
        ]]
    if include_test:
        names += ["results.json"]
    for name in names:
        src, dst = source / name, target / name
        if not src.exists():
            continue
        if dst.exists():
            if ms_hash(src) != ms_hash(dst):
                raise ValueError(f"Previously copied input changed: {dst}")
        else:
            shutil.copy2(src, dst)


def ms_mean(runs, data_dir, output_dir, split):
    runs, output = list(map(Path, runs)), Path(output_dir)
    configs = [ms_read(run / "config.json") for run in runs]
    seeds = [config["seed"] for config in configs]
    if seeds != list(MS_SEEDS):
        raise ValueError(f"Expected the predeclared seed order {MS_SEEDS}, got {seeds}")
    if any(ms_learning_config(config) != ms_learning_config(configs[0]) for config in configs[1:]):
        raise ValueError("Seed averaging requires the same learning setup")
    frame = load_split(data_dir, split)
    arrays = [load_aligned(run, split, frame) for run in runs]
    mean = np.mean(arrays, axis=0)
    prepared = add_context(frame, configs[0]["n_context"], configs[0]["with_speaker"],
                           sep=" [SEP] ", n_future=configs[0]["n_future"])
    config = {"model_name": "DeBERTa-v3-base mean of seeds 42,123,2025",
              "seeds": seeds, "member_runs": list(map(str, runs)),
              "mean_weights": [1 / 3] * 3, "selection_split": "dev"}
    if (output / "config.json").exists() and ms_read(output / "config.json") != config:
        raise ValueError("Seed-mean directory belongs to a different experiment")
    write_json(output / "config.json", config)
    results = ms_read(output / "results.json") if (output / "results.json").exists() else {"model_name": config["model_name"]}
    if (output / f"{split}_probs.npy").exists():
        previous = load_aligned(output, split, frame)
        if not np.allclose(previous, mean, rtol=0, atol=1e-7):
            raise ValueError("Previously saved seed average changed")
    else:
        results[split] = save_split(output, split, prepared, mean)
        write_json(output / "results.json", results)
    return results


def ms_f1(labels, prediction):
    matrix = np.bincount(7 * labels + prediction, minlength=49).reshape(7, 7)
    support = matrix.sum(1)
    denominator = support + matrix.sum(0)
    score = np.divide(2 * np.diag(matrix), denominator, out=np.zeros(7), where=denominator != 0)
    return float((score * support).sum() / support.sum())


def ms_freeze(roberta, mean_run, all_seed_runs, data_dir, output_dir, plan):
    roberta, mean_run, output = Path(roberta), Path(mean_run), Path(output_dir)
    frame = load_split(data_dir, "dev")
    members = [roberta, mean_run]
    probabilities = [load_aligned(run, "dev", frame) for run in members]
    protected = {}
    for run in [*map(Path, all_seed_runs), *members]:
        for name in ["config.json", "dev_probs.npy", "dev_labels.npy", "dev_ids.json", "dev_manifest.json"]:
            path = run / name
            protected[str(path)] = ms_hash(path)
    selection_path = output / "selection.json"
    guard_path = output / "frozen_protocol.json"
    if guard_path.exists():
        guard = ms_read(guard_path)
        if guard["artifact_hashes"] != protected or guard["plan"] != plan or ms_hash(selection_path) != guard["selection_sha256"]:
            raise ValueError("Frozen multi-seed experiment changed; use a new version")
        return ms_read(selection_path)
    best, key, rows = None, None, []
    y = frame.label.to_numpy()
    for weight in MS_GRID["roberta_weights"]:
        logits = np.log(np.clip(weight * probabilities[0] + (1 - weight) * probabilities[1], 1e-12, 1))
        for bias in MS_GRID["neutral_biases"]:
            shifted = logits.copy()
            shifted[:, 0] += bias
            score = ms_f1(y, shifted.argmax(1))
            candidate = (score, -((weight - .5)**2 + (bias - .5)**2), -abs(bias), -weight)
            rows.append({"roberta_weight": weight, "deberta_mean_weight": 1 - weight,
                         "neutral_bias": bias, "dev_weighted_f1": score})
            if key is None or candidate > key:
                key, best = candidate, rows[-1]
    selection = {
        "runs": list(map(str, members)), "weights": [best["roberta_weight"], best["deberta_mean_weight"]],
        "neutral_bias": best["neutral_bias"], "dev_weighted_f1": best["dev_weighted_f1"],
        "grid": MS_GRID, "seeds": list(MS_SEEDS), "seed_mean_rule": "equal mean of all three predeclared seeds",
        "selection_split": "dev", "original_test_previously_inspected": True,
    }
    write_json(selection_path, selection)
    import pandas as pd
    pd.DataFrame(rows).to_csv(output / "dev_search.csv", index=False)
    write_json(output / "results.json", {
        "model_name": "RoBERTa + 3-seed DeBERTa mean",
        "dev": save_split(output, "dev", frame, blend(probabilities, selection["weights"], selection["neutral_bias"])),
    })
    write_json(guard_path, {"plan": plan, "artifact_hashes": protected, "selection_sha256": ms_hash(selection_path)})
    return selection


def ms_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-base")
    parser.add_argument("--output-dir")
    parser.add_argument("--data-dir", default="/content/drive/MyDrive/diploma-mer/data/meld")
    parser.add_argument("--dev-only", action="store_true")
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("Connect a T4 GPU runtime before starting these training runs")
    print("GPU:", torch.cuda.get_device_name(0), "| seeds to train:", MS_EXTRA_SEEDS, flush=True)
    project = Path("/content/drive/MyDrive/diploma-mer")
    old = Path(args.v1_base) if args.v1_base else locate_run(project, "v1")
    out = Path(args.output_dir) if args.output_dir else locate_run(project, "multiseed_v3")
    data = Path(args.data_dir)
    if out.resolve() == old.resolve() or out.resolve().is_relative_to(old.resolve()):
        raise ValueError("Use a new sibling output folder; v1 must remain unchanged")
    source42 = old / "deberta_s42"
    template = ms_read(source42 / "config.json")
    if template.get("seed") != 42 or template.get("model_name") != "microsoft/deberta-v3-base":
        raise ValueError("The reference run must be the completed DeBERTa seed 42")
    if template.get("limit_train") or template.get("limit_eval"):
        raise ValueError("Reference must use the full train/dev splits")
    protocol = ms_read(old / "frozen_protocol.json")
    old_hashes = dict(protocol["artifact_hashes"])
    for path in [old / "frozen_protocol.json", old / "ensemble/selection.json",
                 source42 / "results.json", source42 / "trainer_state.json"]:
        old_hashes[str(path)] = ms_hash(path)
    def check_old():
        for path, digest in old_hashes.items():
            if ms_hash(path) != digest:
                raise ValueError(f"Original v1 artifact changed: {path}")
    check_old()
    plan = {"seeds": list(MS_SEEDS), "new_seeds": list(MS_EXTRA_SEEDS), "learning_config": ms_learning_config(template),
            "original_artifacts": old_hashes, "grid": MS_GRID,
            "protocol": "Train new seeds; equal seed mean; select blend on dev; freeze; evaluate test once",
            "original_test_previously_inspected": True}
    if (out / "experiment_plan.json").exists() and ms_read(out / "experiment_plan.json") != plan:
        raise ValueError("This output folder belongs to another plan")
    write_json(out / "experiment_plan.json", plan)
    reused = out / "reused_inputs"
    for name in ["deberta_s42", "roberta_recheck", "tfidf"]:
        ms_snapshot(old / name, reused / name)
    runs = [reused / "deberta_s42"]
    for seed in MS_EXTRA_SEEDS:
        run = out / f"deberta_s{seed}"
        config = {**template, "seed": seed, "data_dir": str(data), "output_dir": str(run),
                  "checkpoint_dir": f"/content/checkpoints/multiseed_v3/deberta_s{seed}",
                  "evaluate_test": False, "save_model": True, "save_probs": True}
        config_path = out / "configs" / f"deberta_s{seed}.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        completed = (run / "results.json").exists() and (run / "best_model/config.json").exists()
        if completed:
            actual = ms_read(run / "config.json")
            if (ms_learning_config(actual) != ms_learning_config(config)
                    or actual["seed"] != seed
                    or Path(actual["output_dir"]).resolve() != run.resolve()):
                raise ValueError(f"Different configuration in completed run: {run}")
            print(f"REUSING completed seed {seed}", flush=True)
        else:
            print(f"\n===== START SEED {seed} =====", flush=True)
            ms_stream("mer.text.train", ["--config", config_path], out / f"seed_{seed}_console.log")
        score = ms_read(run / "results.json")["dev"]["weighted_f1"]
        print(f"===== FINISHED SEED {seed}: dev weighted F1 = {score:.6f} =====", flush=True)
        check_old()
        runs.append(run)
    mean_run = out / "deberta_seed_mean"
    ms_mean(runs, data, mean_run, "dev")
    ensemble = out / "ensemble_roberta_deberta_mean"
    selection = ms_freeze(reused / "roberta_recheck", mean_run, runs, data, ensemble, plan)
    print("\nFROZEN DEV SELECTION:", json.dumps(selection, indent=2), flush=True)
    if args.dev_only:
        print("Dev-only mode: no new test evaluation. Rerun without --dev-only after freeze.", flush=True)
        return
    for run in runs[1:]:
        if "test" not in ms_read(run / "results.json"):
            ms_stream("mer.evaluation.inference", [
                "--checkpoint", run / "best_model", "--config", run / "config.json",
                "--data-dir", data, "--output-dir", run, "--splits", "test", "--batch-size", "8",
            ], out / f"{run.name}_evaluation.log")
    for name in ["deberta_s42", "roberta_recheck", "tfidf"]:
        ms_snapshot(old / name, reused / name, include_test=True)
    ms_mean(runs, data, mean_run, "test")
    combined = ms_read(ensemble / "results.json")
    if "test" not in combined:
        # Check the frozen dev artifacts again before loading test probabilities.
        ms_freeze(reused / "roberta_recheck", mean_run, runs, data, ensemble, plan)
        frame = load_split(data, "test")
        probabilities = [load_aligned(Path(run), "test", frame) for run in selection["runs"]]
        combined["test"] = save_split(ensemble, "test", frame,
                                      blend(probabilities, selection["weights"], selection["neutral_bias"]))
        combined["selection"] = selection
        write_json(ensemble / "results.json", combined)
    reports = [reused / "tfidf", reused / "roberta_recheck", *runs, mean_run, ensemble]
    for run in reports:
        make_report(run, output_dir=out / "reports" / run.name)
    table = summarize(reports, out / "comparison")
    check_old()
    evidence_zip = out.parent / (out.name + "_evidence.zip")
    with zipfile.ZipFile(evidence_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in out.rglob("*"):
            if path.is_file() and "best_model" not in path.parts and path.suffix in {
                    ".json", ".csv", ".png", ".md", ".npy", ".yaml", ".log", ".sha256"}:
                archive.write(path, path.relative_to(out))
    print(table.to_string(index=False), flush=True)
    print("\nDONE. Test ensemble weighted F1:", combined["test"]["weighted_f1"], flush=True)
    print("Reports:", out, "\nEvidence ZIP:", evidence_zip, flush=True)


if __name__ == "__main__":
    try:
        _ms_in_notebook = get_ipython() is not None
    except NameError:
        _ms_in_notebook = False
    _ms_demo = globals().get("DEMO") if _ms_in_notebook else None
    if _ms_demo is not None and hasattr(_ms_demo, "model"):
        _ms_demo.model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
    try:
        ms_main([] if _ms_in_notebook else None)
    finally:
        if _ms_demo is not None and hasattr(_ms_demo, "model"):
            _ms_demo.model.to(_ms_demo.device)
