"""Refinement must use dev only, preserve its source and respect the frozen choice."""
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import pytest

from mer.data.meld import EMOTIONS, LABEL2ID
from mer.evaluation.artifacts import fingerprint, metrics, sample_ids
from mer.evaluation.ensemble import blend
from mer.evaluation.refine import evaluate, read_frozen, select


def make_evidence(path, include_test=False, bad_ids=False):
    runs = ["roberta_recheck", "deberta_s42"]
    members = {}
    df = pd.DataFrame({
        "dialogue_id": [0, 0, 1, 1, 2, 2, 3], "utterance_id": [0, 1, 0, 1, 0, 1, 0],
        "speaker": ["Speaker"] * 7, "utterance": [f"example {i}" for i in range(7)],
        "label": list(range(7)), "gold": EMOTIONS,
    })
    probabilities = [np.eye(7) * .9 + .1 / 7, np.roll(np.eye(7), 1, axis=1) * .6 + .4 / 7]
    def add_array(name, array):
        buffer = io.BytesIO()
        np.save(buffer, array)
        members[name] = buffer.getvalue()
    def add_json(name, value):
        members[name] = json.dumps(value).encode("utf-8")
    for run, probs in zip(runs, probabilities):
        add_json(f"{run}/config.json", {"model_name": run, "seed": 42})
        for split in ["dev"] + (["test"] if include_test else []):
            add_array(f"{run}/{split}_probs.npy", probs)
            add_array(f"{run}/{split}_labels.npy", df.label.to_numpy())
            ids = sample_ids(df)
            if bad_ids and run == runs[1]:
                ids[0], ids[1] = ids[1], ids[0]
            add_json(f"{run}/{split}_ids.json", ids)
            add_json(f"{run}/{split}_manifest.json", {
                "split": split, "n_examples": len(df), "label_order": EMOTIONS,
                "dataset_fingerprint": fingerprint(df),
            })
            members[f"{run}/{split}_predictions.csv"] = df.to_csv(index=False).encode("utf-8")
    root = "/content/drive/MyDrive/diploma-mer/experiments/v1"
    reference = blend(probabilities, [.5, .5], .5)
    add_json("ensemble/selection.json", {
        "runs": [f"{root}/{run}" for run in runs], "weights": [.5, .5], "neutral_bias": .5,
        "dev_weighted_f1": metrics(df.label.to_numpy(), reference)["weighted_f1"],
    })
    original_hashes = {
        f"{root}/{name}": hashlib.sha256(value).hexdigest()
        for name, value in members.items() if name.endswith(("/config.json", "/dev_probs.npy"))
    }
    add_json("frozen_protocol.json", {
        "best_new": f"{root}/{runs[1]}", "artifact_hashes": original_hashes,
        "ensemble_selection_sha256": hashlib.sha256(members["ensemble/selection.json"]).hexdigest(),
    })
    with zipfile.ZipFile(path, "w") as z:
        for name, payload in members.items():
            z.writestr(name, payload)
    return df


def test_selection_without_any_test_members_is_repeatable(tmp_path):
    archive = tmp_path / "evidence.zip"
    make_evidence(archive)
    before = archive.read_bytes()
    out = tmp_path / "selected"
    selected = select(archive, out)
    assert selected["test_used_for_selection"] is False
    assert selected["candidate_count"] == 651
    assert selected["dev_weighted_f1"] >= selected["reference_dev_weighted_f1"]
    assert not list(out.glob("test*"))
    assert select(archive, out) == selected
    assert archive.read_bytes() == before
    assert read_frozen(out) == selected


def test_misaligned_ids_are_rejected(tmp_path):
    archive = tmp_path / "bad_ids.zip"
    make_evidence(archive, bad_ids=True)
    with pytest.raises(ValueError, match="identities"):
        select(archive, tmp_path / "selected")


def test_frozen_evaluation_and_unchanged_source(tmp_path):
    archive = tmp_path / "evidence.zip"
    make_evidence(archive, include_test=True)
    before = archive.read_bytes()
    out = tmp_path / "selected"
    selected = select(archive, out)
    result = evaluate(archive, out)
    assert result["selection_sha256"] == hashlib.sha256((out / "selection.json").read_bytes()).hexdigest()
    assert "refined_test" in result and result["no_training_required"]
    assert result["paired_dialogue_bootstrap"]["resamples"] == 2000
    assert (out / "comparison.csv").exists()
    assert evaluate(archive, out) == result
    assert archive.read_bytes() == before
    (out / "selection.json").write_text(json.dumps({**selected, "neutral_bias": 99}), encoding="utf-8")
    with pytest.raises(ValueError, match="modified"):
        evaluate(archive, out)


def test_changed_archive_is_rejected_after_selection(tmp_path):
    archive = tmp_path / "evidence.zip"
    make_evidence(archive, include_test=True)
    out = tmp_path / "selected"
    select(archive, out)
    with zipfile.ZipFile(archive, "a") as z:
        z.writestr("extra.json", "{}")
    with pytest.raises(ValueError, match="differs"):
        evaluate(archive, out)
