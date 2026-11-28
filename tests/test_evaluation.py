"""Meaningful protocol/integration checks; no large weights or GPU required."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mer.data.meld import EMOTIONS, ID2LABEL, LABEL2ID
from mer.evaluation.artifacts import load_aligned, metrics, save_split
from mer.evaluation.baseline import fit_baseline
from mer.evaluation.ensemble import blend, select_blend
from mer.evaluation.report import make_report, select_examples


def frame():
    return pd.DataFrame({
        "dialogue_id": list(range(7)), "utterance_id": [0] * 7,
        "speaker": ["Speaker"] * 7, "utterance": [f"emotion {i}" for i in range(7)],
        "label": list(range(7)), "emotion": EMOTIONS, "sentiment": ["neutral"] * 7,
    })


def probs():
    return np.eye(7) * .94 + np.ones((7, 7)) * (.06 / 7)


def test_artifacts_keep_ids_and_reject_reordered_same_label_rows(tmp_path):
    df = pd.concat([frame(), frame().assign(dialogue_id=lambda x: x.dialogue_id + 7)], ignore_index=True)
    p = np.concatenate([probs(), probs()])
    score = save_split(tmp_path, "test", df, p)
    assert score["weighted_f1"] == 1
    assert np.allclose(load_aligned(tmp_path, "test", df), p)
    reordered = df.copy()
    reordered.iloc[[0, 7]] = reordered.iloc[[7, 0]].to_numpy()
    with pytest.raises(ValueError, match="identities"):
        load_aligned(tmp_path, "test", reordered)


def test_invalid_probabilities_fail():
    with pytest.raises(ValueError):
        metrics(np.arange(7), np.full((7, 7), np.nan))
    with pytest.raises(ValueError):
        blend([probs()], [2.0])


def test_baseline_and_dev_selection():
    x = [f"common token{i} emotion{i}" for i in range(7) for _ in range(4)]
    y = np.repeat(np.arange(7), 4)
    model, candidates, _ = fit_baseline((x, y), (x, y), c_grid=[1.0], max_features=1000)
    assert model.predict_proba(x).shape == (28, 7)
    assert len(candidates) == 2
    assert np.array_equal(model.predict(x), y)


def test_ensemble_selection_and_report(tmp_path):
    p = probs()
    wrong = np.roll(p, 1, axis=1)
    selected = select_blend(np.arange(7), [p, wrong])
    assert selected["dev_weighted_f1"] == 1
    score = save_split(tmp_path, "test", frame(), p)
    (tmp_path / "results.json").write_text(json.dumps({"test": score}), encoding="utf-8")
    (tmp_path / "trainer_state.json").write_text(json.dumps({"log_history": [
        {"epoch": .5, "step": 1, "loss": 2.0},
        {"epoch": 1, "step": 2, "eval_loss": 1.5, "eval_weighted_f1": .5},
    ]}), encoding="utf-8")
    out = make_report(tmp_path)
    assert (out / "test_confusion_matrix.png").exists()
    assert (out / "learning_curves.png").exists()
    assert pd.read_csv(out / "test_per_class.csv").support.sum() == 7
    examples = select_examples(pd.read_csv(tmp_path / "test_predictions.csv"))
    assert examples.correct.all()


def test_local_deberta_training_export_and_inference_without_test(monkeypatch, tmp_path):
    import torch
    from transformers import BertTokenizerFast, DebertaV2Config, DebertaV2ForSequenceClassification
    from mer.text import train
    from mer.evaluation.inference import Predictor
    from mer.evaluation import inference

    torch.set_num_threads(1)
    checkpoint = tmp_path / "tiny"
    checkpoint.mkdir()
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "speaker", "emotion", "context", ":"] + [
        f"token{i}" for i in range(7)
    ]
    (checkpoint / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(checkpoint / "vocab.txt"))
    tokenizer.save_pretrained(checkpoint)
    cfg = DebertaV2Config(
        vocab_size=len(vocab), hidden_size=24, num_hidden_layers=1,
        num_attention_heads=4, intermediate_size=32, max_position_embeddings=64,
        relative_attention=True, position_buckets=16, max_relative_positions=32,
        num_labels=7, id2label=ID2LABEL, label2id=LABEL2ID,
    )
    DebertaV2ForSequenceClassification(cfg).save_pretrained(checkpoint)
    df = frame().assign(utterance=[f"token{i}" for i in range(7)])
    def load(_, split):
        assert split != "test", "Test must remain untouched during model selection"
        return df
    monkeypatch.setattr(train, "download_meld", lambda _: None)
    monkeypatch.setattr(train, "load_split", load)
    output = tmp_path / "output"
    config = tmp_path / "config.yaml"
    config.write_text(
        f"model_name: {checkpoint.as_posix()}\noutput_dir: {output.as_posix()}\n"
        f"checkpoint_dir: {(tmp_path / 'checkpoints').as_posix()}\n"
        "epochs: 1\nbatch_size: 2\ngrad_accum: 2\nmax_length: 32\nn_context: 0\n"
        "evaluate_test: false\nsave_model: true\ngradient_checkpointing: true\nlogging_steps: 1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(config)])
    train.main()
    result = json.loads((output / "results.json").read_text(encoding="utf-8"))
    assert "test" not in result
    assert result["hardware"]["parameters"] > 0
    assert (output / "trainer_state.json").exists()
    predictor = Predictor(output / "best_model", output / "config.json", device="cpu")
    from mer.text.encoding import encode_example
    boundary = encode_example(tokenizer, "token1 " * 29, "token2", max_length=32)
    assert len(boundary["input_ids"]) <= 32
    with pytest.raises(ValueError, match="architecture"):
        Predictor(output / "best_model", {**predictor.cfg, "model_name": "FacebookAI/roberta-large"}, device="cpu")
    prediction = predictor.predict("token1 " * 200, context="Speaker: token2", speaker="Speaker")
    assert prediction["sequence_length"] <= 32
    assert len(prediction["token_ids"]) == len(prediction["attention_mask"])
    assert abs(sum(prediction["scores"].values()) - 1) < 1e-5
    with pytest.raises(ValueError):
        predictor.predict("  ")

    # Final evaluation must preserve the frozen config/dev files and training evidence.
    frozen = {name: (output / name).read_bytes() for name in ["config.json", "dev_probs.npy"]}
    monkeypatch.setattr(inference, "download_meld", lambda _: None)
    monkeypatch.setattr(inference, "load_split", lambda _, split: df)
    monkeypatch.setattr(sys, "argv", [
        "inference", "--checkpoint", str(output / "best_model"), "--config", str(output / "config.json"),
        "--output-dir", str(output), "--splits", "test", "--benchmark-repeats", "2",
    ])
    inference.main()
    after = json.loads((output / "results.json").read_text(encoding="utf-8"))
    assert "dev" in after and "test" in after and "train_minutes" in after
    assert "training_hardware" in after
    for name, data in frozen.items():
        assert (output / name).read_bytes() == data


def test_notebook_freeze_cell_is_repeatable_and_detects_changed_selection(monkeypatch, tmp_path):
    import hashlib
    from mer.data.meld import SPLIT_FILES
    from mer.evaluation.artifacts import write_json
    from mer.evaluation import ensemble

    df = frame()
    data = tmp_path / "data"
    data.mkdir()
    raw = df.rename(columns={
        "dialogue_id": "Dialogue_ID", "utterance_id": "Utterance_ID",
        "speaker": "Speaker", "utterance": "Utterance", "emotion": "Emotion", "sentiment": "Sentiment",
    })
    raw.to_csv(data / SPLIT_FILES["dev"], index=False)
    baseline = tmp_path / "baseline"
    runs = [tmp_path / "seed42", tmp_path / "seed123"]
    for run in [baseline, *runs]:
        score = save_split(run, "dev", df, probs())
        write_json(run / "config.json", {"model_name": "fixture", "seed": 42})
        write_json(run / "results.json", {"dev": score})
    notebook = json.loads((Path(__file__).parents[1] / "notebooks/colab_deberta_meld.ipynb").read_text(encoding="utf-8"))
    source = next("".join(cell["source"]) for cell in notebook["cells"]
                  if cell["cell_type"] == "code" and "".join(cell["source"]).startswith("BEST_NEW = max("))
    def run_module(module, *args):
        assert module == "mer.evaluation.ensemble"
        monkeypatch.setattr(sys, "argv", ["ensemble", *map(str, args)])
        ensemble.main()
    namespace = dict(
        NEW_RUNS=runs, ROBERTA_RUN=tmp_path/"missing", BASELINE=baseline, BASE=tmp_path/"notebook",
        DATA=data, SEEDS=[42, 123], LEGACY_PROB_RUNS=[], ALLOW_LEGACY_ORDER=False,
        Path=Path, json=json, hashlib=hashlib, write_json=write_json, run_module=run_module,
    )
    exec(compile(source, "notebook_freeze", "exec"), namespace)
    frozen = namespace["freeze_file"].read_bytes()
    exec(compile(source, "notebook_freeze", "exec"), namespace)
    assert namespace["freeze_file"].read_bytes() == frozen
    selection = namespace["ENSEMBLE"] / "selection.json"
    selection.write_bytes(selection.read_bytes() + b"\n")
    with pytest.raises(RuntimeError, match="Artifacts differ"):
        exec(compile(source, "notebook_freeze", "exec"), namespace)
