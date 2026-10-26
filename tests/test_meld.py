import pandas as pd

from mer.data.meld import EMOTIONS, LABEL2ID, add_context


def make_df():
    return pd.DataFrame(
        {
            "dialogue_id": [0, 0, 0, 1, 1],
            "utterance_id": [0, 1, 2, 0, 1],
            "speaker": ["Ross", "Rachel", "Ross", "Joey", "Chandler"],
            "utterance": ["Hi.", "Hey!", "How are you?", "How you doin'?", "Could I BE any more tired?"],
            "emotion": ["neutral", "joy", "neutral", "joy", "sadness"],
            "label": [0, 1, 0, 1, 4],
        }
    )


def test_label_order_is_fixed():
    assert EMOTIONS[0] == "neutral" and len(EMOTIONS) == 7
    assert LABEL2ID["joy"] == 1


def test_context_stays_inside_dialogue():
    df = add_context(make_df(), n_context=2, sep=" | ")
    assert df.loc[0, "context"] == ""
    assert df.loc[2, "context"] == "Ross: Hi. | Rachel: Hey!"
    # first turn of dialogue 1 must not see dialogue 0
    assert df.loc[3, "context"] == ""
    assert df.loc[4, "context"] == "Joey: How you doin'?"
    assert df.loc[4, "target"] == "Chandler: Could I BE any more tired?"


def test_no_context():
    df = add_context(make_df(), n_context=0, with_speaker=False)
    assert (df["context"] == "").all()
    assert df.loc[1, "target"] == "Hey!"
