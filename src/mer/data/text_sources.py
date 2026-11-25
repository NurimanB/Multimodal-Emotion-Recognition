"""Real (human-written) text emotion corpora mapped to the 7 MELD emotions.

Every loader returns a DataFrame with the same columns (see COLUMNS):

    text, emotion, lang, source, split, orig_label, weak,
    dialogue_id, utterance_id, speaker

- emotion is one of mer.data.meld.EMOTIONS, or None if the original label has
  no clear counterpart (those rows are dropped by the builder).
- weak=True marks labels that were not given by human annotators per text
  (CARER: labels come from the author's own hashtags).
- dialogue sources (MELD, DailyDialog, EmoryNLP) keep dialogue_id/utterance_id
  so the previous turns can be used as context, the same way as for MELD.

Machine-translated corpora (ru_go_emotions, ru-izard-emotions,
ru_twitter_emotions, ...) and LLM-generated corpora are deliberately not
included: the supervisor asked for real data only.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pandas as pd

from mer.data.meld import EMOTIONS, download_meld, load_split

COLUMNS = [
    "text", "emotion", "lang", "source", "split", "orig_label", "weak",
    "dialogue_id", "utterance_id", "speaker",
]

HF_PARQUET = "https://huggingface.co/api/datasets/{repo}/parquet/{config}/{split}/0.parquet"


# ---------------------------------------------------------------- label maps

# Official GoEmotions -> Ekman grouping (google-research/goemotions, ekman_mapping.json).
GOEMOTIONS_NAMES = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral",
]
GOEMOTIONS_TO_EKMAN = {
    "anger": "anger", "annoyance": "anger", "disapproval": "anger",
    "disgust": "disgust",
    "fear": "fear", "nervousness": "fear",
    "joy": "joy", "amusement": "joy", "approval": "joy", "excitement": "joy",
    "gratitude": "joy", "love": "joy", "optimism": "joy", "relief": "joy", "pride": "joy",
    "admiration": "joy", "desire": "joy", "caring": "joy",
    "sadness": "sadness", "disappointment": "sadness", "embarrassment": "sadness",
    "grief": "sadness", "remorse": "sadness",
    "surprise": "surprise", "realization": "surprise", "confusion": "surprise",
    "curiosity": "surprise",
    "neutral": "neutral",
}

# DailyDialog emotion ids (Li et al., 2017).
DAILYDIALOG_NAMES = ["no emotion", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]
DAILYDIALOG_MAP = {
    "no emotion": "neutral", "anger": "anger", "disgust": "disgust", "fear": "fear",
    "happiness": "joy", "sadness": "sadness", "surprise": "surprise",
}

# EmoryNLP (Zahiri & Choi, 2018) uses Willcox's feeling wheel. Peaceful and
# Powerful have no counterpart among the 7 classes and are dropped by default.
EMORYNLP_MAP = {
    "Joyful": "joy", "Mad": "anger", "Sad": "sadness", "Scared": "fear", "Neutral": "neutral",
    "Peaceful": None, "Powerful": None,
}

# CARER (Saravia et al., 2018): tweets labelled by their own emotion hashtags.
CARER_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]
CARER_MAP = {"sadness": "sadness", "joy": "joy", "love": "joy", "anger": "anger",
             "fear": "fear", "surprise": "surprise"}

# CEDR (Sboev et al., 2021), Russian. No labels = "no emotion".
CEDR_NAMES = ["joy", "sadness", "surprise", "fear", "anger"]

# BRIGHTER / SemEval-2025 Task 11 track A: six binary emotion columns.
BRIGHTER_EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

# IEMOCAP transcripts (licensed; only read from a local file the user provides).
# Frustration and "other" have no counterpart and are dropped.
IEMOCAP_MAP = {
    "ang": "anger", "hap": "joy", "exc": "joy", "sad": "sadness", "neu": "neutral",
    "sur": "surprise", "fea": "fear", "dis": "disgust", "fru": None, "oth": None, "xxx": None,
}


# ---------------------------------------------------------------- helpers

def _download(url: str, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not (target.exists() and target.stat().st_size > 0):
        print(f"Downloading {url} ...")
        tmp = target.with_suffix(target.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(target)
    return target


def _hf_parquet(cache: Path, repo: str, config: str, split: str) -> pd.DataFrame:
    """Read one split of a Hugging Face dataset from its auto-converted parquet file."""
    path = cache / repo.replace("/", "__") / config / f"{split}.parquet"
    _download(HF_PARQUET.format(repo=repo, config=config, split=split), path)
    return pd.read_parquet(path)


def _single(labels: set) -> str | None:
    """Keep a multi-label example only if all its labels map to one emotion."""
    labels = {x for x in labels if x is not None}
    return next(iter(labels)) if len(labels) == 1 else None


def _frame(rows: list[dict] | pd.DataFrame, **const) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for key, value in const.items():
        df[key] = value
    for col in COLUMNS:
        if col not in df:
            df[col] = None
    df["weak"] = df["weak"].fillna(False).astype(bool)
    return df[COLUMNS]


# ---------------------------------------------------------------- English

def load_meld(cache: Path) -> pd.DataFrame:
    data_dir = download_meld(cache / "meld")
    parts = []
    for split in ["train", "dev", "test"]:
        df = load_split(data_dir, split)
        parts.append(_frame({
            "text": df["utterance"],
            "emotion": df["emotion"],
            "orig_label": df["emotion"],
            "dialogue_id": "meld:" + split + ":" + df["dialogue_id"].astype(str),
            "utterance_id": df["utterance_id"],
            "speaker": df["speaker"],
        }, lang="en", source="meld", split=split))
    return pd.concat(parts, ignore_index=True)


def load_emorynlp(cache: Path) -> pd.DataFrame:
    base = "https://raw.githubusercontent.com/emorynlp/emotion-detection/master/json"
    rows = []
    for split, name in [("train", "trn"), ("dev", "dev"), ("test", "tst")]:
        path = _download(f"{base}/emotion-detection-{name}.json", cache / "emorynlp" / f"{name}.json")
        season = json.loads(path.read_text(encoding="utf-8"))
        for episode in season["episodes"]:
            for scene in episode["scenes"]:
                for i, utt in enumerate(scene["utterances"]):
                    rows.append({
                        "text": utt["transcript"],
                        "orig_label": utt["emotion"],
                        "emotion": EMORYNLP_MAP.get(utt["emotion"]),
                        "split": split,
                        "dialogue_id": f"emorynlp:{scene['scene_id']}",
                        "utterance_id": i,
                        "speaker": ", ".join(utt["speakers"]),
                    })
    return _frame(rows, lang="en", source="emorynlp")


def load_goemotions(cache: Path) -> pd.DataFrame:
    base = "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data"
    parts = []
    for split, name in [("train", "train"), ("dev", "dev"), ("test", "test")]:
        path = _download(f"{base}/{name}.tsv", cache / "goemotions" / f"{name}.tsv")
        df = pd.read_csv(path, sep="\t", header=None, names=["text", "labels", "id"],
                         quoting=3, dtype=str, keep_default_na=False)
        names = [[GOEMOTIONS_NAMES[int(i)] for i in s.split(",")] for s in df["labels"]]
        parts.append(_frame({
            "text": df["text"],
            "orig_label": [",".join(n) for n in names],
            "emotion": [_single({GOEMOTIONS_TO_EKMAN[x] for x in n}) for n in names],
        }, lang="en", source="goemotions", split=split))
    return pd.concat(parts, ignore_index=True)


def load_dailydialog(cache: Path) -> pd.DataFrame:
    """DailyDialog from the parquet copy of roskoN/dailydialog (original splits)."""
    rows = []
    for split, hf_split in [("train", "train"), ("dev", "validation"), ("test", "test")]:
        df = _hf_parquet(cache, "roskoN/dailydialog", "full", hf_split)
        utt_col = next(c for c in ["utterances", "dialog", "dialogue"] if c in df)
        emo_col = next(c for c in ["emotions", "emotion"] if c in df)
        id_col = next((c for c in ["id", "dialog_id"] if c in df), None)
        for d, rec in df.iterrows():
            did = rec[id_col] if id_col else d
            for i, (text, emo) in enumerate(zip(rec[utt_col], rec[emo_col])):
                name = DAILYDIALOG_NAMES[int(emo)] if str(emo).isdigit() else str(emo)
                rows.append({
                    "text": text, "orig_label": name, "emotion": DAILYDIALOG_MAP.get(name),
                    "split": split, "dialogue_id": f"dailydialog:{split}:{did}",
                    "utterance_id": i, "speaker": "A" if i % 2 == 0 else "B",
                })
    return _frame(rows, lang="en", source="dailydialog")


def load_carer(cache: Path) -> pd.DataFrame:
    """CARER, the full 417k version. It has no official split; the builder makes one."""
    df = _hf_parquet(cache, "dair-ai/emotion", "unsplit", "train")
    names = [CARER_NAMES[int(i)] for i in df["label"]]
    return _frame({
        "text": df["text"], "orig_label": names, "emotion": [CARER_MAP[n] for n in names],
    }, lang="en", source="carer", split="train", weak=True)


def _load_brighter(cache: Path, config: str, lang: str) -> pd.DataFrame:
    parts = []
    for split in ["train", "dev", "test"]:
        df = _hf_parquet(cache, "brighter-dataset/BRIGHTER-emotion-categories", config, split)
        df = df.dropna(subset=BRIGHTER_EMOTIONS)
        on = [[e for e in BRIGHTER_EMOTIONS if int(r[e]) > 0] for _, r in df.iterrows()]
        parts.append(_frame({
            "text": df["text"].values,
            "orig_label": [",".join(x) or "none" for x in on],
            # No emotion marked = neutral (BRIGHTER has no explicit neutral class).
            "emotion": [_single(set(x)) if x else "neutral" for x in on],
        }, lang=lang, source=f"brighter_{config}", split=split))
    return pd.concat(parts, ignore_index=True)


def load_brighter_eng(cache: Path) -> pd.DataFrame:
    return _load_brighter(cache, "eng", "en")


def load_iemocap(cache: Path, path: str | None = None) -> pd.DataFrame:
    """IEMOCAP transcripts from a local CSV with columns text, emotion (and optionally
    split, dialogue_id, utterance_id, speaker). IEMOCAP needs a signed licence, so
    it is never downloaded automatically."""
    if not path:
        raise FileNotFoundError("iemocap needs iemocap_csv=<path to local CSV>")
    df = pd.read_csv(path)
    df["orig_label"] = df["emotion"]
    df["emotion"] = df["emotion"].str.lower().str[:3].map(IEMOCAP_MAP)
    if "split" not in df:
        df["split"] = "train"
    return _frame(df, lang="en", source="iemocap")


# ---------------------------------------------------------------- Russian

def load_cedr(cache: Path) -> pd.DataFrame:
    parts = []
    for split in ["train", "test"]:
        df = _hf_parquet(cache, "sagteam/cedr_v1", "main", split)
        names = [[CEDR_NAMES[int(i)] for i in labels] for labels in df["labels"]]
        parts.append(_frame({
            "text": df["text"].values,
            "orig_label": [",".join(n) or "no_emotion" for n in names],
            "emotion": [_single(set(n)) if n else "neutral" for n in names],
        }, lang="ru", source="cedr", split=split))
    return pd.concat(parts, ignore_index=True)


def load_brighter_rus(cache: Path) -> pd.DataFrame:
    return _load_brighter(cache, "rus", "ru")


# ---------------------------------------------------------------- Kazakh

def load_kk_annotated(cache: Path, path: str | None = None) -> pd.DataFrame:
    """Our own Kazakh set, labelled by hand (see docs/annotation_guide_kk.md).
    CSV columns: text, emotion, split (train/dev/test)."""
    if not path:
        raise FileNotFoundError("kk_annotated needs kk_csv=<path to labelled CSV>")
    df = pd.read_csv(path)
    df["orig_label"] = df["emotion"]
    df["emotion"] = df["emotion"].str.lower().str.strip()
    return _frame(df, lang="kk", source="kk_annotated")


LOADERS = {
    "meld": load_meld,
    "emorynlp": load_emorynlp,
    "goemotions": load_goemotions,
    "dailydialog": load_dailydialog,
    "carer": load_carer,
    "brighter_eng": load_brighter_eng,
    "iemocap": load_iemocap,
    "cedr": load_cedr,
    "brighter_rus": load_brighter_rus,
    "kk_annotated": load_kk_annotated,
}

# Sources that need a local file; key = config option holding the path.
LOCAL_SOURCES = {"iemocap": "iemocap_csv", "kk_annotated": "kk_csv"}

assert all(v in EMOTIONS for m in [GOEMOTIONS_TO_EKMAN, DAILYDIALOG_MAP, CARER_MAP]
           for v in m.values())
