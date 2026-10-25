import pandas as pd

from mer.data.meld import EMOTIONS, LABEL2ID


def test_label_order_is_fixed():
    assert EMOTIONS[0] == "neutral" and len(EMOTIONS) == 7
    assert LABEL2ID["joy"] == 1
