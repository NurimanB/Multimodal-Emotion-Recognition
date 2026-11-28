"""Existing run locations must remain usable after project renaming."""
import json
from pathlib import Path
import runpy

import pytest

from mer.evaluation.paths import locate_run


def make_v1(path):
    (path / "deberta_s42").mkdir(parents=True)
    (path / "deberta_s42/config.json").write_text("{}", encoding="utf-8")
    (path / "frozen_protocol.json").write_text(json.dumps({
        "best_new": str(path / "deberta_s42"), "ensemble_runs": [],
    }), encoding="utf-8")


def test_existing_reference_folder_is_found_without_renaming(tmp_path):
    assert locate_run(tmp_path, "v1") == tmp_path / "experiments/v1"
    old = tmp_path / "recorded_runs/first_complete"
    make_v1(old)
    assert locate_run(tmp_path, "v1") == old
    preferred = tmp_path / "experiments/v1"
    make_v1(preferred)
    assert locate_run(tmp_path, "v1") == preferred


def test_multiseed_discovery_rejects_ambiguous_recordings(tmp_path):
    for name in ["first", "second"]:
        run = tmp_path / "recorded_runs" / name
        run.mkdir(parents=True)
        (run / "experiment_plan.json").write_text(json.dumps({
            "seeds": [42, 123, 2025], "new_seeds": [123, 2025],
        }), encoding="utf-8")
        if name == "first":
            assert locate_run(tmp_path, "multiseed_v3") == run
    with pytest.raises(ValueError, match="Multiple"):
        locate_run(tmp_path, "multiseed_v3")


def test_inference_entry_point_has_all_run_paths_without_loading_weights():
    script = Path(__file__).resolve().parents[1] / "scripts/demo_ensemble.py"
    namespace = runpy.run_path(str(script), run_name="ensemble_import_check")
    assert namespace["REFERENCE_RUN"].name == "v1"
    assert namespace["DEMO_RUN"].name == "multiseed_v3"
    assert "FrozenEnsemblePredictor" in namespace
    assert "ENSEMBLE_DEMO" not in namespace
