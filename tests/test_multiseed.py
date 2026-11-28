"""Regression checks for the additional-seed coordinator; no GPU training."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mer.data.meld import EMOTIONS, load_split
from mer.evaluation.artifacts import load_aligned, save_split, write_json
from mer.text.train import DEFAULTS

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_deberta_multiseed.py"
spec = importlib.util.spec_from_file_location("multiseed_launcher", SCRIPT)
ms = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ms)


def dataset(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    for split in ("train", "dev", "test"):
        pd.DataFrame({
            "Dialogue_ID": [0] * 7, "Utterance_ID": list(range(7)),
            "Speaker": ["Speaker"] * 7, "Utterance": [f"example {i}" for i in range(7)],
            "Emotion": EMOTIONS, "Sentiment": ["neutral"] * 7,
        }).to_csv(data / f"{split}_sent_emo.csv", index=False)
    return data


def config(seed):
    return {**DEFAULTS, "model_name": "microsoft/deberta-v3-base",
            "seed": seed, "n_context": 3, "n_future": 0, "with_speaker": True,
            "evaluate_test": False, "save_model": True, "save_probs": True}


def make_run(path, data, cfg, offset=0, test=True):
    write_json(path / "config.json", cfg)
    result = {"model_name": cfg["model_name"]}
    probabilities = .8 * np.roll(np.eye(7), offset, axis=1) + .2 / 7
    for split in ["dev"] + (["test"] if test else []):
        result[split] = save_split(path, split, load_split(data, split), probabilities)
    write_json(path / "results.json", result)
    return probabilities


def test_equal_mean_requires_exact_seeds_and_same_training(tmp_path):
    data = dataset(tmp_path)
    runs = [tmp_path / f"seed{s}" for s in ms.MS_SEEDS]
    arrays = [make_run(run, data, config(seed), offset=i, test=False)
              for i, (run, seed) in enumerate(zip(runs, ms.MS_SEEDS))]
    output = tmp_path / "mean"
    ms.ms_mean(runs, data, output, "dev")
    actual = load_aligned(output, "dev", load_split(data, "dev"))
    np.testing.assert_allclose(actual, np.mean(arrays, axis=0), atol=1e-7)
    frozen_bytes = (output / "dev_probs.npy").read_bytes()
    ms.ms_mean(runs, data, output, "dev")
    assert (output / "dev_probs.npy").read_bytes() == frozen_bytes
    with pytest.raises(ValueError, match="seed order"):
        ms.ms_mean(list(reversed(runs)), data, output, "dev")
    altered = ms.ms_read(runs[1] / "config.json")
    altered["lr"] = .01
    write_json(runs[1] / "config.json", altered)
    with pytest.raises(ValueError, match="same learning setup"):
        ms.ms_mean(runs, data, output, "dev")


def test_dev_freeze_needs_no_test_and_rejects_modified_artifacts(tmp_path):
    data = dataset(tmp_path)
    runs = [tmp_path / f"seed{s}" for s in ms.MS_SEEDS]
    for run, seed in zip(runs, ms.MS_SEEDS):
        make_run(run, data, config(seed), test=False)
    roberta = tmp_path / "roberta"
    make_run(roberta, data, {"model_name": "roberta"}, test=False)
    mean_run = tmp_path / "mean"
    ms.ms_mean(runs, data, mean_run, "dev")
    output = tmp_path / "ensemble"
    selected = ms.ms_freeze(roberta, mean_run, runs, data, output, {"seeds": list(ms.MS_SEEDS)})
    assert selected["selection_split"] == "dev"
    assert selected["seeds"] == [42, 123, 2025]
    assert len(pd.read_csv(output / "dev_search.csv")) == 651
    assert not list(tmp_path.rglob("test_probs.npy"))
    assert ms.ms_freeze(roberta, mean_run, runs, data, output, {"seeds": list(ms.MS_SEEDS)}) == selected
    with (runs[0] / "dev_manifest.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="Frozen"):
        ms.ms_freeze(roberta, mean_run, runs, data, output, {"seeds": list(ms.MS_SEEDS)})


def test_snapshot_copies_dev_first_and_preserves_source(tmp_path):
    data = dataset(tmp_path)
    source, destination = tmp_path / "source", tmp_path / "destination"
    make_run(source, data, config(42))
    before = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    ms.ms_snapshot(source, destination)
    assert (destination / "dev_probs.npy").exists()
    assert not (destination / "test_probs.npy").exists()
    assert not (destination / "results.json").exists()
    ms.ms_snapshot(source, destination, include_test=True)
    assert (destination / "test_probs.npy").exists()
    assert (destination / "results.json").exists()
    assert before == {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}


def test_full_coordinator_trains_only_two_new_seeds_and_reuses_completed(tmp_path, monkeypatch):
    data = dataset(tmp_path)
    old, out = tmp_path / "v1", tmp_path / "v3"
    for name in ("deberta_s42", "roberta_recheck", "tfidf"):
        make_run(old / name, data, config(42))
    write_json(old / "deberta_s42/trainer_state.json", {"global_step": 1875})
    write_json(old / "ensemble/selection.json", {"weights": [.5, .5], "neutral_bias": .5})
    protected = {str(p): ms.ms_hash(p) for p in old.rglob("*") if p.is_file()}
    write_json(old / "frozen_protocol.json", {"artifact_hashes": protected})
    before = {str(p): p.read_bytes() for p in old.rglob("*") if p.is_file()}
    calls = []

    def fake_stream(module, arguments, logfile):
        import yaml
        arguments = list(map(str, arguments))
        calls.append((module, arguments))
        get = lambda flag: arguments[arguments.index(flag) + 1]
        if module == "mer.text.train":
            cfg = yaml.safe_load(Path(get("--config")).read_text())
            assert cfg["seed"] in [123, 2025]
            assert cfg["evaluate_test"] is False
            assert isinstance(cfg["lr"], float)
            cfg["checkpoint_dir"] = str(tmp_path / "recorded_checkpoints" / str(cfg["seed"]))
            run = Path(cfg["output_dir"])
            make_run(run, data, cfg, test=False)
            write_json(run / "best_model/config.json", {"model_type": "deberta-v2"})
        elif module == "mer.evaluation.inference":
            run = Path(get("--output-dir"))
            assert (out / "ensemble_roberta_deberta_mean/frozen_protocol.json").is_file()
            result = ms.ms_read(run / "results.json")
            result["test"] = save_split(run, "test", load_split(data, "test"), .8 * np.eye(7) + .2 / 7)
            write_json(run / "results.json", result)
        else:
            pytest.fail(f"Unexpected subprocess: {module}")

    monkeypatch.setattr(ms.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(ms.torch.cuda, "get_device_name", lambda _: "test fixture")
    monkeypatch.setattr(ms, "ms_stream", fake_stream)
    # Rendering is independently covered; this check exercises orchestration and caching.
    monkeypatch.setattr(ms, "make_report", lambda *args, **kwargs: None)
    args = ["--v1-base", str(old), "--output-dir", str(out), "--data-dir", str(data)]
    ms.ms_main(args)
    assert [c[0] for c in calls] == ["mer.text.train"] * 2 + ["mer.evaluation.inference"] * 2
    assert ms.ms_read(out / "ensemble_roberta_deberta_mean/results.json")["test"]["weighted_f1"] == 1
    assert (out.parent / (out.name + "_evidence.zip")).is_file()
    assert before == {str(p): p.read_bytes() for p in old.rglob("*") if p.is_file()}
    calls.clear()
    ms.ms_main(args)
    assert calls == []
    assert before == {str(p): p.read_bytes() for p in old.rglob("*") if p.is_file()}
