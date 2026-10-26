"""MELD (Multimodal EmotionLines Dataset) text loading.

Only the text CSVs are used here. The raw video/audio archive (~10 GB) is
needed later for the audio and video branches.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

BASE_URL = "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD"
SPLIT_FILES = {
    "train": "train_sent_emo.csv",
    "dev": "dev_sent_emo.csv",
    "test": "test_sent_emo.csv",
}

# Fixed label order so label ids are the same in every run and every branch.
EMOTIONS = ["neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear"]
LABEL2ID = {name: i for i, name in enumerate(EMOTIONS)}
ID2LABEL = {i: name for name, i in LABEL2ID.items()}


def download_meld(data_dir: str | Path) -> Path:
    """Download the three MELD text CSVs into data_dir (skips existing files)."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for fname in SPLIT_FILES.values():
        target = data_dir / fname
        if target.exists() and target.stat().st_size > 0:
            continue
        print(f"Downloading {fname} ...")
        urllib.request.urlretrieve(f"{BASE_URL}/{fname}", target)
    return data_dir


def _fix_text(text: str) -> str:
    """MELD utterances contain mojibake like 'Iâ€™m'. Repair it."""
    try:
        import ftfy

        return ftfy.fix_text(text)
    except ImportError:
        try:
            return text.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return text


def load_split(data_dir: str | Path, split: str) -> pd.DataFrame:
    """Load one split as a DataFrame sorted by dialogue and utterance order."""
    path = Path(data_dir) / SPLIT_FILES[split]
    df = pd.read_csv(path)
    df = df.rename(
        columns={
            "Utterance": "utterance",
            "Speaker": "speaker",
            "Emotion": "emotion",
            "Sentiment": "sentiment",
            "Dialogue_ID": "dialogue_id",
            "Utterance_ID": "utterance_id",
        }
    )
    df["utterance"] = df["utterance"].astype(str).map(_fix_text).str.strip()
    df["speaker"] = df["speaker"].astype(str).map(_fix_text).str.strip()
    df["emotion"] = df["emotion"].str.lower().str.strip()
    df["label"] = df["emotion"].map(LABEL2ID)
    if df["label"].isna().any():
        bad = sorted(df.loc[df["label"].isna(), "emotion"].unique())
        raise ValueError(f"Unknown emotion labels in {split}: {bad}")
    df["label"] = df["label"].astype(int)
    df = df.sort_values(["dialogue_id", "utterance_id"]).reset_index(drop=True)
    return df[["dialogue_id", "utterance_id", "speaker", "utterance", "emotion", "sentiment", "label"]]


def add_context(
    df: pd.DataFrame, n_context: int, with_speaker: bool = True, sep: str = " </s> "
) -> pd.DataFrame:
    """Add a 'context' column with the previous n_context turns of the same dialogue.

    The target utterance itself is kept in 'target'. With n_context=0 the
    context is empty and the model sees only the utterance.
    """
    df = df.copy()

    def fmt(row) -> str:
        return f"{row.speaker}: {row.utterance}" if with_speaker else row.utterance

    df["target"] = [fmt(r) for r in df.itertuples()]
    contexts = []
    for _, group in df.groupby("dialogue_id", sort=False):
        turns = group["target"].tolist()
        for i in range(len(turns)):
            start = max(0, i - n_context)
            contexts.append(sep.join(turns[start:i]) if n_context > 0 else "")
    df["context"] = contexts
    return df


def class_counts(df: pd.DataFrame) -> pd.Series:
    return df["emotion"].value_counts().reindex(EMOTIONS, fill_value=0)
