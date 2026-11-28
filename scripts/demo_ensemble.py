"""Final frozen four-checkpoint ensemble demo for an already configured Colab."""
from pathlib import Path
import gc
import hashlib
import json
import time

import numpy as np
import torch

from mer.data.meld import EMOTIONS
from mer.evaluation.artifacts import benchmark, write_json
from mer.evaluation.ensemble import blend
from mer.evaluation.inference import Predictor, launch_widget
from mer.evaluation.paths import locate_run

PROJECT_ROOT = Path("/content/drive/MyDrive/diploma-mer")
DEMO_ROOT = PROJECT_ROOT / "experiments"
DEMO_RUN = locate_run(PROJECT_ROOT, "multiseed_v3")
REFERENCE_RUN = locate_run(PROJECT_ROOT, "v1")


class FrozenEnsemblePredictor:
    def __init__(self):
        if not torch.cuda.is_available():
            raise RuntimeError("Use the T4 runtime for the four-checkpoint demo")
        selection_path = DEMO_RUN / "ensemble_roberta_deberta_mean/selection.json"
        self.selection = json.loads(selection_path.read_text())
        guard = json.loads((selection_path.parent / "frozen_protocol.json").read_text())
        if hashlib.sha256(selection_path.read_bytes()).hexdigest() != guard["selection_sha256"]:
            raise ValueError("The frozen selection was changed")
        if self.selection["seeds"] != [42, 123, 2025]:
            raise ValueError("Unexpected ensemble seeds")
        paths = [
            (Path("/content/drive/MyDrive/diploma-mer/outputs/text_roberta_large_ctx3/saved_model"),
             DEMO_RUN / "reused_inputs/roberta_recheck/config.json"),
            (REFERENCE_RUN / "deberta_s42/best_model",
             DEMO_RUN / "reused_inputs/deberta_s42/config.json"),
            (DEMO_RUN / "deberta_s123/best_model", DEMO_RUN / "deberta_s123/config.json"),
            (DEMO_RUN / "deberta_s2025/best_model", DEMO_RUN / "deberta_s2025/config.json"),
        ]
        self.members = []
        for checkpoint, config in paths:
            print("Loading demo checkpoint:", checkpoint, flush=True)
            self.members.append(Predictor(checkpoint, config))
        self.model = self.members[1].model
        self.device = self.members[1].device
        self.parameters = sum(sum(p.numel() for p in member.model.parameters()) for member in self.members)

    def predict(self, text, context="", speaker="Speaker"):
        outputs = [member.predict(text, context, speaker) for member in self.members]
        arrays = [np.array([[output["scores"][emotion] for emotion in EMOTIONS]])
                  for output in outputs]
        probabilities = blend([arrays[0], np.mean(arrays[1:], axis=0)],
                              self.selection["weights"], self.selection["neutral_bias"])[0]
        result = dict(outputs[1])
        result.update(
            emotion=EMOTIONS[int(probabilities.argmax())],
            confidence=float(probabilities.max()),
            scores={emotion: float(p) for emotion, p in zip(EMOTIONS, probabilities)},
            confidence_note="Uncalibrated blended score after neutral bias; tokens shown for DeBERTa seed 42",
            model="Frozen RoBERTa + three-seed DeBERTa ensemble",
            weights=self.selection["weights"], neutral_bias=self.selection["neutral_bias"],
            token_view="DeBERTa-v3-base seed 42; RoBERTa uses a different tokenizer",
        )
        return result


if __name__ == "__main__":
    # Remove the old demo from GPU while loading the final ensemble.
    previous_demo = globals().get("DEMO")
    if previous_demo is not None and hasattr(previous_demo, "model"):
        previous_demo.model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    ENSEMBLE_DEMO = globals().get("ENSEMBLE_DEMO") or FrozenEnsemblePredictor()
    load_seconds = time.perf_counter() - started
    prompts = [
        "I am so happy to see you!",
        "I cannot believe you did that!",
        "The meeting starts at six.",
    ]
    examples = [ENSEMBLE_DEMO.predict(prompt, speaker="Rachel") for prompt in prompts]
    latency = benchmark(
        lambda: ENSEMBLE_DEMO.predict(prompts[0], speaker="Rachel"),
        warmup=3, repeats=30, synchronize=torch.cuda.synchronize,
    )
    evidence = {
        "purpose": "Typed live-demo prompts, not a held-out evaluation",
        "examples": examples,
        "latency": latency,
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "parameters": ENSEMBLE_DEMO.parameters,
            "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "load_seconds": load_seconds,
            "scope": "Four checkpoints simultaneously resident; sequential batch-1 inference; current Colab session",
        },
        "selection": ENSEMBLE_DEMO.selection,
        "confidence_note": "Uncalibrated score; no claim of correctness without ground truth",
    }
    destination = DEMO_ROOT / "inference_demo/demo_evidence.json"
    write_json(destination, evidence)
    print("DEMO_READY", json.dumps({"p50_ms": latency["p50_ms"],
          "p95_ms": latency["p95_ms"], "throughput": latency["throughput_examples_per_second"],
          "first_example": {k: examples[0][k] for k in ["input", "emotion", "confidence", "sequence_length"]}}),
          flush=True)
    launch_widget(ENSEMBLE_DEMO)
    from google.colab import files
    files.download(str(destination))
