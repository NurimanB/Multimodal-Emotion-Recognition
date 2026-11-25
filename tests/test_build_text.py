import pandas as pd

from mer.data.build_text import carve_split, clean_text, dedupe, norm_key
from mer.data.meld import EMOTIONS
from mer.data.text_sources import (
    COLUMNS, DAILYDIALOG_MAP, EMORYNLP_MAP, GOEMOTIONS_NAMES, GOEMOTIONS_TO_EKMAN, _single,
)


def rows(*items):
    df = pd.DataFrame(items, columns=["text", "emotion", "source", "split"])
    for col in COLUMNS:
        if col not in df:
            df[col] = None
    df["weak"] = False
    return df


def test_label_maps_cover_targets():
    assert set(GOEMOTIONS_TO_EKMAN) == set(GOEMOTIONS_NAMES)
    assert set(GOEMOTIONS_TO_EKMAN.values()) == set(EMOTIONS)
    assert set(DAILYDIALOG_MAP.values()) == set(EMOTIONS)
    assert {v for v in EMORYNLP_MAP.values() if v} <= set(EMOTIONS)


def test_single_label_rule():
    assert _single({"joy", "joy"}) == "joy"
    assert _single({"joy", "anger"}) is None
    assert _single({"joy", None}) == "joy"


def test_clean_and_norm():
    assert clean_text("Look  at https://x.com/a @bob ") == "Look at <url> @user"
    assert norm_key("Oh my God!!  Really?") == "oh my god really"


def test_meld_is_protected_and_leaks_removed():
    df = rows(
        ("I can not believe you did that", "anger", "meld", "test"),
        ("I can't believe you did that", "anger", "meld", "test"),
        ("I can not believe you did that!", "anger", "emorynlp", "train"),
        ("Okay.", "neutral", "meld", "test"),
        ("Okay.", "neutral", "meld", "train"),
        ("i can not believe you did that", "joy", "meld", "train"),
    )
    kept, removed = dedupe(df, ["meld", "emorynlp"], min_words=4)
    # both MELD test rows stay, even the near-duplicate pair
    assert ((kept["source"] == "meld") & (kept["split"] == "test")).sum() == 3
    # the EmoryNLP copy of a MELD test sentence is a leak
    assert list(removed["reason"]) == ["leak_test"]
    # MELD's own train copy of a test sentence is kept (benchmark unchanged)
    assert ((kept["source"] == "meld") & (kept["split"] == "train")).sum() == 2
    # short utterances are not deduplicated
    assert (kept["text"] == "Okay.").sum() == 2


def test_label_conflict_in_train_drops_all_copies():
    df = rows(
        ("this is the best day ever", "joy", "goemotions", "train"),
        ("This is the best day ever!", "surprise", "goemotions", "train"),
        ("what a lovely little morning", "joy", "goemotions", "train"),
        ("What a lovely little morning", "joy", "goemotions", "train"),
    )
    kept, removed = dedupe(df, ["goemotions"], min_words=4)
    assert list(kept["text"]) == ["what a lovely little morning"]
    assert sorted(removed["reason"]) == ["duplicate", "label_conflict", "label_conflict"]


def test_carve_split_is_stratified():
    df = rows(*[(f"text {i}", "joy" if i % 2 else "anger", "carer", "train") for i in range(200)])
    out = carve_split(df, 0.1, "test", seed=0)
    test = out[out["split"] == "test"]
    assert len(test) == 20 and (test["emotion"] == "joy").sum() == 10
