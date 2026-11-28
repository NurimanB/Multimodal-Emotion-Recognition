"""Load a saved MELD classifier for evaluation or a token-visible demo."""
from __future__ import annotations
import argparse
import contextlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from mer.data.meld import EMOTIONS, _fix_text, add_context, download_meld, load_split
from mer.evaluation.artifacts import benchmark, runtime_info, save_split, write_json
from mer.text.encoding import encode_example


def read_config(path):
    if path:
        config_path = Path(path)
        text = config_path.read_text(encoding="utf-8")
        value = json.loads(text) if config_path.suffix.lower() == ".json" else (yaml.safe_load(text) or {})
    else:
        value = {}
    return {"n_context": 3, "n_future": 0, "with_speaker": True, "max_length": 256, **value}


class Predictor:
    def __init__(self, checkpoint, config=None, device=None):
        self.cfg = read_config(config) if not isinstance(config, dict) else config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.tokenizer.truncation_side = "left"
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(self.device).eval()
        labels = [self.model.config.id2label.get(i) for i in range(7)]
        if self.model.config.num_labels != 7 or labels != EMOTIONS:
            raise ValueError(f"Checkpoint labels {labels} differ from MELD {EMOTIONS}; verify checkpoint provenance")
        name = str(self.cfg.get("model_name", "")).lower()
        expected_type = ("xlm-roberta" if "xlm-roberta" in name else
                         "roberta" if "roberta" in name else
                         "deberta-v2" if "deberta-v3" in name else None)
        if expected_type and self.model.config.model_type != expected_type:
            raise ValueError(f"Checkpoint architecture {self.model.config.model_type} differs from {expected_type}")
        self.checkpoint = str(checkpoint)

    def _forward(self, features):
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt").to(self.device)
        amp = torch.autocast("cuda", dtype=torch.float16) if self.device.type == "cuda" else contextlib.nullcontext()
        with torch.inference_mode(), amp:
            logits = self.model(**batch).logits
        return torch.softmax(logits.float(), dim=-1).cpu().numpy()

    def predict(self, text, context="", speaker="Speaker"):
        cleaned = _fix_text(str(text)).strip()
        if not cleaned:
            raise ValueError("Enter nonempty text")
        target = f"{speaker.strip() or 'Speaker'}: {cleaned}" if self.cfg.get("with_speaker", True) else cleaned
        turns = [line.strip() for line in context.splitlines() if line.strip()]
        n_context = int(self.cfg.get("n_context", 3))
        context = f" {self.tokenizer.sep_token} ".join(turns[-n_context:]) if n_context else ""
        encoded = encode_example(self.tokenizer, target, context, int(self.cfg.get("max_length", 256)))
        probs = self._forward([encoded])[0]
        return {
            "input": text, "cleaned": cleaned, "target": target, "context": context,
            "tokens": self.tokenizer.convert_ids_to_tokens(encoded["input_ids"]),
            "token_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"],
            "sequence_length": len(encoded["input_ids"]), "max_length": int(self.cfg.get("max_length", 256)),
            "emotion": EMOTIONS[int(probs.argmax())], "confidence": float(probs.max()),
            "scores": {name: float(p) for name, p in zip(EMOTIONS, probs)},
            "confidence_note": "Uncalibrated softmax score; not certainty of correctness",
            "future_context_used_in_demo": False,
        }

    def predict_frame(self, df, batch_size=16):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        prepared = add_context(df, int(self.cfg.get("n_context", 3)),
                               self.cfg.get("with_speaker", True),
                               sep=f" {self.tokenizer.sep_token} ", n_future=int(self.cfg.get("n_future", 0)))
        predictions = []
        for start in range(0, len(prepared), batch_size):
            chunk = prepared.iloc[start:start + batch_size]
            features = [encode_example(self.tokenizer, row.target, row.context,
                                       int(self.cfg.get("max_length", 256))) for row in chunk.itertuples()]
            predictions.append(self._forward(features))
        return prepared, np.concatenate(predictions)


def launch_widget(predictor):
    import ipywidgets as widgets
    from IPython.display import clear_output, display
    text = widgets.Textarea(value="I cannot believe you did that!", description="Text:", layout={"width": "95%"})
    context = widgets.Textarea(value="", description="Previous:", placeholder="One turn per line, with speakers",
                               layout={"width": "95%"})
    speaker = widgets.Text(value="Speaker", description="Speaker:")
    button = widgets.Button(description="Predict emotion", button_style="primary")
    output = widgets.Output()

    def clicked(_):
        with output:
            clear_output()
            try:
                result = predictor.predict(text.value, context.value, speaker.value)
                print(f"Emotion: {result['emotion']} | confidence: {result['confidence']:.3f}")
                print(result["confidence_note"])
                for key in ["cleaned", "target", "context", "tokens", "token_ids", "attention_mask", "sequence_length"]:
                    print(f"{key}: {result[key]}")
                import pandas as pd
                display(pd.DataFrame(result["scores"].items(), columns=["emotion", "softmax_score"]))
            except (ValueError, RuntimeError) as error:
                print(error)
    button.on_click(clicked)
    display(widgets.VBox([text, context, speaker, button, output]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config")
    parser.add_argument("--text")
    parser.add_argument("--context", default="")
    parser.add_argument("--speaker", default="Speaker")
    parser.add_argument("--data-dir", default="data/meld")
    parser.add_argument("--output-dir")
    parser.add_argument("--splits", nargs="+", choices=["dev", "test"], default=["dev", "test"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--benchmark-repeats", type=int, default=30)
    args = parser.parse_args()
    predictor = Predictor(args.checkpoint, args.config)
    if args.text is not None:
        print(json.dumps(predictor.predict(args.text, args.context, args.speaker), ensure_ascii=False, indent=2))
        return
    if not args.output_dir:
        parser.error("--output-dir is required for evaluation")
    download_meld(args.data_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config_file = out / "config.json"
    if config_file.exists():
        if json.loads(config_file.read_text(encoding="utf-8")) != predictor.cfg:
            raise ValueError("Output directory already belongs to a different run configuration")
    else:
        write_json(config_file, predictor.cfg)
    previous = out / "results.json"
    result = json.loads(previous.read_text(encoding="utf-8")) if previous.exists() else {}
    if result.get("hardware") and "training_hardware" not in result:
        result["training_hardware"] = result["hardware"]
    result.update(checkpoint=args.checkpoint, model_name=predictor.cfg.get("model_name", predictor.model.config.model_type),
                  evaluation_protocol="Checkpoint/config fixed before this held-out evaluation")
    if predictor.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for split in args.splits:
        frame, probs = predictor.predict_frame(load_split(args.data_dir, split), args.batch_size)
        result[split] = save_split(out, split, frame, probs)
        print(f"{split} weighted F1: {result[split]['weighted_f1']:.4f}")
    sync = torch.cuda.synchronize if predictor.device.type == "cuda" else lambda: None
    result["hardware"] = runtime_info(predictor.model)
    result["hardware"]["latency"] = benchmark(
        lambda: predictor.predict("I cannot believe you did that!", speaker="Rachel"),
        repeats=args.benchmark_repeats, synchronize=sync,
    )
    result["hardware"]["evaluation_batch_size"] = args.batch_size
    result["hardware"]["max_length"] = predictor.cfg.get("max_length", 256)
    write_json(out / "results.json", result)


if __name__ == "__main__":
    main()
