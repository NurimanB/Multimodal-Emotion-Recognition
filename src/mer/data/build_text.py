"""Build the pooled text emotion dataset (EN/RU/KK real corpora -> 7 MELD emotions).

Usage:
    python -m mer.data.build_text --config configs/data_text_pilot.yaml
    python -m mer.data.build_text --config configs/data_text_v1.yaml sources=[meld,cedr]

Output in out_dir: train/dev/test.parquet (+ .csv if write_csv), stats.json and
stats.md with counts per source, language and class, and what was removed.

Rules:
- Only human-written text. Rows whose label has no counterpart among the 7
  emotions, or multi-label rows whose labels map to different emotions, are dropped.
- Official splits are kept. Sources without a dev/test split get one carved
  from train (stratified by emotion, fixed seed).
- MELD is never changed (train/dev/test stay identical to the benchmark).
  Exact duplicates (after normalisation) of texts with at least
  min_words_dedup words are removed from the other sources: a MELD copy wins,
  then test over dev over train, then earlier sources over later ones. Inside
  train, a text that appears with different emotions is dropped entirely.
  Short texts ("Okay.", "What?") are left alone: in dialogues they are
  separate examples, not leaks.
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from mer.data.meld import EMOTIONS, LABEL2ID
from mer.data.text_sources import LOADERS, LOCAL_SOURCES

DEFAULTS = {
    "sources": ["meld", "emorynlp", "dailydialog", "goemotions", "brighter_eng", "carer",
                "cedr", "brighter_rus"],
    "cache_dir": "data/raw",
    "out_dir": "data/text_v1",
    "seed": 42,
    "dev_frac": 0.05,  # carved from train when a source has no dev split
    "test_frac": 0.05,  # carved from train when a source has no test split
    "min_words_dedup": 4,
    "max_chars": 1000,
    "include_weak": True,  # CARER (hashtag labels); kept, but flagged weak=True
    "limit_per_source": None,  # pilot: at most N train rows per source (dev/test: N // 5)
    "skip_failed": False,  # continue if one source fails to download
    "write_csv": False,
    "iemocap_csv": None,
    "kk_csv": None,
}

SPLITS = ["train", "dev", "test"]
URL_RE = re.compile(r"https?://\S+|www\.\S+")
USER_RE = re.compile(r"(?<!\w)@\w+")
SPACE_RE = re.compile(r"\s+")
NORM_RE = re.compile(r"[^\w\s]")


def parse_config(defaults=DEFAULTS, argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*", help="key=value pairs")
    args = parser.parse_args(argv)
    cfg = dict(defaults)
    cfg.update(yaml.safe_load(Path(args.config).read_text()) or {})
    for item in args.overrides:
        key, value = item.split("=", 1)
        if key not in defaults:
            raise KeyError(f"Unknown config key: {key}")
        cfg[key] = yaml.safe_load(value)
    return cfg


def clean_text(text) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = URL_RE.sub("<url>", text)
    text = USER_RE.sub("@user", text)
    return SPACE_RE.sub(" ", text).strip()


def norm_key(text: str) -> str:
    """Key used to find duplicates: lower case, no punctuation, single spaces."""
    return SPACE_RE.sub(" ", NORM_RE.sub(" ", text.lower())).strip()


def carve_split(df: pd.DataFrame, frac: float, new_split: str, seed: int) -> pd.DataFrame:
    """Move a stratified frac of a source's train rows into new_split."""
    df = df.copy()
    train = df[df["split"] == "train"]
    picked = (train.groupby("emotion", group_keys=False)
              .apply(lambda g: g.sample(frac=frac, random_state=seed)).index)
    df.loc[picked, "split"] = new_split
    return df


def limit_source(df: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    """Pilot sampling. Dialogue sources are sampled by whole dialogues."""
    parts = []
    for split, part in df.groupby("split"):
        n = limit if split == "train" else max(1, limit // 5)
        if part["dialogue_id"].notna().all():
            ids = part["dialogue_id"].drop_duplicates().sample(frac=1.0, random_state=seed)
            sizes = part.groupby("dialogue_id").size().reindex(ids)
            keep = sizes.index[(sizes.cumsum() <= n) | (np.arange(len(sizes)) == 0)]
            parts.append(part[part["dialogue_id"].isin(keep)])
        else:
            parts.append(part.sample(n=min(n, len(part)), random_state=seed))
    return pd.concat(parts)


def dedupe(df: pd.DataFrame, source_order: list[str], min_words: int,
           protected: tuple[str, ...] = ("meld",)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove cross-split leaks and duplicates. Returns (kept, removed with a reason).

    Rows of protected sources (MELD) are never removed, so MELD stays identical
    to the published benchmark; other sources lose any copy of a MELD text.
    """
    df = df.copy()
    df["key"] = df["text"].map(norm_key)
    long = df["key"].str.split().str.len() >= min_words
    is_protected = df["source"].isin(protected)
    split_rank = df["split"].map({"test": 0, "dev": 1, "train": 2})
    source_rank = df["source"].map({s: i for i, s in enumerate(source_order)})
    df["rank"] = (split_rank * 1000 + source_rank + 1).where(~is_protected, -1)
    df["split_rank"] = split_rank

    removed = []
    cand = df[long]
    owner = cand.groupby("key")["rank"].transform("min")
    not_owner = cand.index[(cand["rank"] != owner) & ~is_protected[long]]
    owner_split = (cand[cand["rank"] == owner].sort_values("split_rank")
                   .drop_duplicates("key").set_index("key")["split"])
    owner_split = cand.loc[not_owner, "key"].map(owner_split)
    # Same split as the owner = plain duplicate; otherwise a leak into dev/test.
    reason = np.where(owner_split.values == cand.loc[not_owner, "split"].values,
                      "duplicate", "leak_" + owner_split.astype(str).values)
    removed.append(df.loc[not_owner].assign(reason=reason))
    df = df.drop(index=not_owner)

    # Same text, same source and split: conflicting labels in train -> drop all,
    # otherwise keep the first copy.
    cand = df[(df["key"].str.split().str.len() >= min_words) & ~df["source"].isin(protected)]
    n_labels = cand.groupby(["source", "split", "key"])["emotion"].transform("nunique")
    conflict = cand.index[(n_labels > 1) & (cand["split"] == "train")]
    removed.append(df.loc[conflict].assign(reason="label_conflict"))
    df = df.drop(index=conflict)
    cand = cand.drop(index=conflict)
    dup = cand.index[cand.duplicated(["source", "split", "key"], keep="first")]
    removed.append(df.loc[dup].assign(reason="duplicate"))
    df = df.drop(index=dup)
    drop_cols = ["key", "rank", "split_rank"]
    return df.drop(columns=drop_cols), pd.concat(removed).drop(columns=drop_cols)


def build(cfg: dict) -> dict:
    cache = Path(cfg["cache_dir"])
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    seed = cfg["seed"]

    frames, failed, dropped = [], {}, []
    for name in cfg["sources"]:
        loader = LOADERS[name]
        try:
            if name in LOCAL_SOURCES:
                df = loader(cache, cfg[LOCAL_SOURCES[name]])
            else:
                df = loader(cache)
        except Exception as exc:  # noqa: BLE001
            if not cfg["skip_failed"]:
                raise
            failed[name] = f"{type(exc).__name__}: {exc}"
            print(f"!! {name} failed: {failed[name]}")
            continue
        if df["weak"].any() and not cfg["include_weak"]:
            continue
        df["text"] = df["text"].map(clean_text)
        bad = df["emotion"].isna() | (df["text"] == "") | (df["text"].str.len() > cfg["max_chars"])
        dropped.append(df[bad].assign(reason=np.where(df.loc[bad, "emotion"].isna(), "unmapped_label",
                                                      "empty_or_too_long")))
        df = df[~bad].copy()
        if not (df["split"] == "test").any():
            df = carve_split(df, cfg["test_frac"], "test", seed)
        if not (df["split"] == "dev").any():
            df = carve_split(df, cfg["dev_frac"] / (1 - cfg["test_frac"]), "dev", seed)
        if cfg["limit_per_source"]:
            df = limit_source(df, int(cfg["limit_per_source"]), seed)
        print(f"{name}: {len(df)} rows")
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    data, removed = dedupe(data, cfg["sources"], cfg["min_words_dedup"])
    dropped.append(removed)
    dropped = pd.concat(dropped, ignore_index=True)
    data["label"] = data["emotion"].map(LABEL2ID).astype(int)

    for split in SPLITS:
        part = data[data["split"] == split].reset_index(drop=True)
        part.to_parquet(out / f"{split}.parquet", index=False)
        if cfg["write_csv"]:
            part.to_csv(out / f"{split}.csv", index=False)

    stats = make_stats(data, dropped, failed, cfg)
    (out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    (out / "stats.md").write_text(stats_markdown(data, dropped, failed))
    print((out / "stats.md").read_text())
    return stats


def _table(df: pd.DataFrame, index: str) -> pd.DataFrame:
    t = pd.crosstab(df[index], df["emotion"]).reindex(columns=EMOTIONS, fill_value=0)
    t["total"] = t.sum(axis=1)
    return t


def make_stats(data, dropped, failed, cfg) -> dict:
    meld_train = int(((data["source"] == "meld") & (data["split"] == "train")).sum())
    train = data[data["split"] == "train"]
    return {
        "config": cfg,
        "rows": {s: int((data["split"] == s).sum()) for s in SPLITS},
        "train_vs_meld_train": round(len(train) / meld_train, 1) if meld_train else None,
        "by_source_split": pd.crosstab(data["source"], data["split"]).to_dict(),
        "train_by_lang_class": _table(train, "lang").to_dict(orient="index"),
        "train_by_source_class": _table(train, "source").to_dict(orient="index"),
        "weak_train_rows": int(train["weak"].sum()),
        "dropped": dropped.groupby(["source", "reason"]).size().unstack(fill_value=0).to_dict(orient="index"),
        "failed_sources": failed,
    }


def stats_markdown(data, dropped, failed) -> str:
    train = data[data["split"] == "train"]
    meld_train = ((data["source"] == "meld") & (data["split"] == "train")).sum()
    lines = ["# Pooled text dataset", ""]
    if meld_train:
        lines += [f"Train rows: {len(train):,} = {len(train) / meld_train:.1f}x MELD train "
                  f"({train['weak'].sum():,} with weak hashtag labels).", ""]
    lines += ["## Rows per source and split", "",
              pd.crosstab(data["source"], data["split"], margins=True).reindex(
                  columns=SPLITS + ["All"]).to_markdown(), "",
              "## Train rows per language and class", "", _table(train, "lang").to_markdown(), "",
              "## Train rows per source and class", "", _table(train, "source").to_markdown(), "",
              "## Removed rows", "",
              dropped.groupby(["source", "reason"]).size().unstack(fill_value=0).to_markdown(), ""]
    if failed:
        lines += ["## Failed sources", ""] + [f"- {k}: {v}" for k, v in failed.items()] + [""]
    return "\n".join(lines)


def main():
    build(parse_config())


if __name__ == "__main__":
    main()
