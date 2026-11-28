"""TF-IDF + Logistic Regression. Fit train only, select on dev, test on request."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from mer.data.meld import add_context, download_meld, load_split
from mer.evaluation.artifacts import benchmark, runtime_info, save_split, write_json


def inputs(df, n_context=3, with_speaker=True):
    prepared = add_context(df, n_context, with_speaker, sep=" </s> ", n_future=0)
    texts = (prepared.target + "\n[context] " + prepared.context).tolist()
    return prepared, texts


def fit_baseline(train, dev, c_grid=(0.5, 2.0, 8.0), seed=42, max_features=50000):
    start = time.perf_counter()
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_features=max_features,
        sublinear_tf=True, lowercase=True, dtype=np.float64,
    )
    x_train = vectorizer.fit_transform(train[0])
    x_dev = vectorizer.transform(dev[0])
    candidates, best, best_score = [], None, -1.0
    for c in c_grid:
        if c <= 0:
            raise ValueError("C must be positive")
        for weight in [None, "balanced"]:
            classifier = LogisticRegression(
                C=c, class_weight=weight, solver="lbfgs", max_iter=1500, random_state=seed,
            )
            classifier.fit(x_train, train[1])
            if not np.array_equal(classifier.classes_, np.arange(7)):
                raise ValueError("Training data must contain all seven MELD classes")
            score = float(f1_score(dev[1], classifier.predict(x_dev), average="weighted"))
            candidates.append({"C": c, "class_weight": weight, "dev_weighted_f1": score})
            if score > best_score:
                best_score = score
                best = classifier
    pipeline = Pipeline([("tfidf", vectorizer), ("classifier", best)])
    return pipeline, candidates, (time.perf_counter() - start) / 60


def evaluate_saved(run_dir, data_dir, splits=("test",)):
    import json
    run = Path(run_dir)
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    pipeline = joblib.load(run / "baseline.joblib")
    results = json.loads((run / "results.json").read_text(encoding="utf-8"))
    for split in splits:
        frame, texts = inputs(load_split(data_dir, split), config["n_context"], config["with_speaker"])
        results[split] = save_split(run, split, frame, pipeline.predict_proba(texts))
    write_json(run / "results.json", results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/meld")
    parser.add_argument("--output-dir", default="outputs/experiments/tfidf")
    parser.add_argument("--n-context", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--c-grid", nargs="+", type=float, default=[0.5, 2, 8])
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()
    download_meld(args.data_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_df = load_split(args.data_dir, "train")
    dev_df = load_split(args.data_dir, "dev")
    train, train_x = inputs(train_df, args.n_context)
    dev, dev_x = inputs(dev_df, args.n_context)
    pipeline, candidates, minutes = fit_baseline(
        (train_x, train.label.to_numpy()), (dev_x, dev.label.to_numpy()), args.c_grid, args.seed,
    )
    joblib.dump(pipeline, out / "baseline.joblib")
    config = {
        "model_name": "TF-IDF + Logistic Regression", "seed": args.seed, "n_context": args.n_context,
        "n_future": 0, "with_speaker": True, "selection_split": "dev",
        "C": pipeline["classifier"].C, "class_weight": pipeline["classifier"].class_weight,
        "ngram_range": [1, 2], "max_features": 50000, "min_df": 2,
        "lowercase": True, "sublinear_tf": True,
    }
    write_json(out / "config.json", config)
    pd.DataFrame(candidates).to_csv(out / "dev_search.csv", index=False)
    results = {"train_minutes": minutes, "model_name": config["model_name"],
               "dev": save_split(out, "dev", dev, pipeline.predict_proba(dev_x))}
    if args.evaluate_test:
        test, test_x = inputs(load_split(args.data_dir, "test"), args.n_context)
        results["test"] = save_split(out, "test", test, pipeline.predict_proba(test_x))
    results["hardware"] = runtime_info()
    results["hardware"]["parameters"] = int(
        pipeline["classifier"].coef_.size + pipeline["classifier"].intercept_.size
    )
    results["hardware"]["latency"] = benchmark(lambda: pipeline.predict_proba([dev_x[0]]))
    write_json(out / "results.json", results)
    print(f"Selected on dev: C={config['C']} class_weight={config['class_weight']}")
    print(f"Dev weighted F1: {results['dev']['weighted_f1']:.4f}")


if __name__ == "__main__":
    main()
