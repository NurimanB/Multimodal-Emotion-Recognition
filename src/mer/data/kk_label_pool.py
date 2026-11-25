"""Make a pool of real Kazakh texts for hand labelling (7 emotions).

Source: KazSAnDRA reviews (issai/kazsandra, gated on Hugging Face: accept the
terms once, then download full/full.csv). Only reviews with extreme scores
(1-2 and 5 stars) are used, because they carry the most emotion.

Rare emotions (fear, disgust, surprise, sadness) are almost absent from a
random sample of reviews, so part of the pool is drawn from texts that contain
cue words for them (lists below, written by hand). No model and no
translation is used to choose or label anything.

Usage:
    python -m mer.data.kk_label_pool --kazsandra data/raw/kazsandra/full.csv \
        --out data/own/kk_pool --size 3000

Writes:
    pool_for_labeling.csv  id, text, annotator_1, annotator_2, emotion, split
                           (shuffled; score and cue bucket are hidden so they
                           do not bias the annotators)
    pool_meta.csv          id, score, domain, bucket, has_kk_letters
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from mer.data.build_text import clean_text, norm_key

# Lower-case stems; a text is in a bucket if it contains any of them.
# Kazakh first, then the Russian words that often appear in mixed reviews.
CUES = {
    "fear": ["қорқ", "үрей", "қауіп", "қауіпт", "алаңда", "уайым", "зәрем", "шошы",
             "страшно", "боюсь", "опасно"],
    "disgust": ["жиіркен", "жексұрын", "сасық", "былғаныш", "лас ", "кір-қожалақ", "құсқым",
                "фу ", "отврат", "мерзк", "противно"],
    "surprise": ["таң қал", "таңқал", "таңғал", "шынымен бе", "күтпеген", "ойпырмай", "мәссаған",
                 "апырай", "ого", "вау", "неожидан"],
    "sadness": ["өкініш", "мұң", "қайғы", "жылад", "көңілім түс", "көңіл қалд", "амал жоқ",
                "жаль", "грустно", "разочар"],
    "anger": ["ашу", "ашулан", "масқара", "алдамшы", "алдады", "ұят", "жынды", "ақшамды қайтар",
              "бесит", "ужас", "позор"],
}
KK_LETTERS = re.compile(r"[әғқңөұүһі]", re.IGNORECASE)


def _col(df: pd.DataFrame, names: list[str]) -> str | None:
    return next((c for c in names if c in df.columns), None)


def load_kazsandra(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    text, score = _col(df, ["text", "review", "content"]), _col(df, ["label", "score", "rating"])
    domain = _col(df, ["domain", "source", "category"])
    if text is None or score is None:
        raise ValueError(f"Cannot find text/score columns in {list(df.columns)}")
    out = pd.DataFrame({
        "text": df[text].map(clean_text),
        "score": pd.to_numeric(df[score], errors="coerce"),
        "domain": df[domain] if domain else "unknown",
    })
    # KazSAnDRA scores may be 0-4 instead of 1-5; shift if so.
    if out["score"].min() == 0 and out["score"].max() == 4:
        out["score"] += 1
    return out


def bucket_of(text: str) -> str:
    low = f" {text.lower()} "
    hits = [emo for emo, stems in CUES.items() if any(s in low for s in stems)]
    return hits[0] if len(hits) == 1 else ("mixed" if hits else "none")


def make_pool(df: pd.DataFrame, size: int, cue_share: float, seed: int,
              min_chars: int = 15, max_chars: int = 400) -> pd.DataFrame:
    df = df[df["score"].isin([1, 2, 5])]
    df = df[df["text"].str.len().between(min_chars, max_chars)]
    df = df.assign(key=df["text"].map(norm_key)).drop_duplicates("key")
    df["bucket"] = df["text"].map(bucket_of)
    df["has_kk_letters"] = df["text"].str.contains(KK_LETTERS)

    # 1) cue part: the same number of texts from each rare-emotion bucket
    n_cue = int(size * cue_share)
    per_bucket = n_cue // len(CUES)
    cue = pd.concat([
        g.sample(n=min(per_bucket, len(g)), random_state=seed)
        for b, g in df[df["bucket"].isin(CUES)].groupby("bucket")
    ]) if n_cue else df.iloc[:0]

    # 2) random part: stratified by score (1, 2, 5) and domain
    rest = df.drop(index=cue.index)
    n_rand = size - len(cue)
    weights = rest.groupby(["score", "domain"])["text"].transform("size")
    rand = rest.sample(n=min(n_rand, len(rest)), weights=1 / weights, random_state=seed)

    pool = pd.concat([cue, rand]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    pool.insert(0, "id", [f"kz{i:05d}" for i in range(len(pool))])
    return pool.drop(columns=["key"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kazsandra", required=True, help="path to KazSAnDRA full.csv")
    parser.add_argument("--out", default="data/own/kk_pool")
    parser.add_argument("--size", type=int, default=3000)
    parser.add_argument("--cue_share", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pool = make_pool(load_kazsandra(args.kazsandra), args.size, args.cue_share, args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sheet = pool[["id", "text"]].assign(annotator_1="", annotator_2="", emotion="", split="")
    sheet.to_csv(out / "pool_for_labeling.csv", index=False)
    pool[["id", "score", "domain", "bucket", "has_kk_letters"]].to_csv(out / "pool_meta.csv", index=False)
    print(f"{len(pool)} texts -> {out}")
    print(pool.groupby(["bucket", "score"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
