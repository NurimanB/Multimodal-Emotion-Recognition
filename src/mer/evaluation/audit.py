"""Dataset summary and duplicate screening; preserve the official MELD splits."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import config_context
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from mer.data.meld import EMOTIONS, download_meld, load_split
from mer.evaluation.artifacts import fingerprint, write_json


def normalize(text):
    return re.sub(r"\s+", " ", str(text).casefold()).strip()


def audit(data_dir, output_dir, near_duplicates=False):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    splits = {name: load_split(data_dir, name) for name in ["train", "dev", "test"]}
    summary, counts = {}, []
    normalized = {}
    for split, df in splits.items():
        normalized[split] = df.utterance.map(normalize)
        summary[split] = {
            "n_examples": len(df), "n_dialogues": int(df.dialogue_id.nunique()),
            "empty_utterances": int(df.utterance.str.strip().eq("").sum()),
            "duplicate_sample_keys": int(df.duplicated(["dialogue_id", "utterance_id"]).sum()),
            "repeated_text_rows": int(normalized[split].duplicated().sum()),
            "dataset_fingerprint": fingerprint(df),
        }
        for emotion in EMOTIONS:
            counts.append({"split": split, "emotion": emotion, "count": int(df.emotion.eq(emotion).sum())})
    summary["exact_text_overlap"] = {}
    for left, right in [("train", "dev"), ("train", "test"), ("dev", "test")]:
        left_texts = set(normalized[left]) - {""}
        summary["exact_text_overlap"][f"{left}/{right}"] = {
            "distinct_shared_texts": len(left_texts & set(normalized[right])),
            "rows_in_right_seen_in_left": int(normalized[right].isin(left_texts).sum()),
        }
    if near_duplicates:
        train_mask = normalized["train"].str.len().ge(20)
        train = splits["train"].loc[train_mask].reset_index(drop=True)
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=30000)
        train_vectors = vectorizer.fit_transform(train.utterance)
        neighbors = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute").fit(train_vectors)
        rows = []
        for split in ["dev", "test"]:
            mask = normalized[split].str.len().ge(20) & ~normalized[split].isin(set(normalized["train"]))
            query = splits[split].loc[mask]
            if query.empty:
                continue
            with config_context(working_memory=64):
                distances, indexes = neighbors.kneighbors(vectorizer.transform(query.utterance))
            for (_, row), distance, index in zip(query.iterrows(), distances[:, 0], indexes[:, 0]):
                similarity = float(1 - distance)
                if similarity >= 0.95:
                    match = train.iloc[int(index)]
                    rows.append({"split": split, "dialogue_id": int(row.dialogue_id),
                                 "utterance_id": int(row.utterance_id), "text": row.utterance,
                                 "train_text": match.utterance, "char_tfidf_cosine": similarity})
        pd.DataFrame(rows, columns=["split", "dialogue_id", "utterance_id", "text",
                                    "train_text", "char_tfidf_cosine"]).to_csv(out / "near_duplicates.csv", index=False)
        summary["near_duplicate_screen"] = {
            "method": "char_wb TF-IDF 3-5 grams; nearest training utterance; cosine >= .95; text >= 20 chars",
            "candidate_pairs": len(rows), "interpretation": "manual-review candidates, not a leakage-free guarantee",
        }
    summary["protocol"] = "Official splits preserved. Vectorizers fit on train; selection on dev; test only after freeze."
    pd.DataFrame(counts).to_csv(out / "class_distribution.csv", index=False)
    write_json(out / "dataset_audit.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/meld")
    parser.add_argument("--output-dir", default="outputs/experiments/dataset")
    parser.add_argument("--near-duplicates", action="store_true")
    args = parser.parse_args()
    download_meld(args.data_dir)
    print(audit(args.data_dir, args.output_dir, args.near_duplicates))


if __name__ == "__main__":
    main()
