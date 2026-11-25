import pandas as pd

from mer.data.kk_label_pool import bucket_of, make_pool


def test_bucket_of():
    assert bucket_of("Өте қорқынышты болды") == "fear"
    assert bucket_of("Жиіркенішті, ешкімге ұсынбаймын") == "disgust"
    assert bucket_of("Рахмет, бәрі жақсы") == "none"
    assert bucket_of("Локация ашу үшін телеграм жазамыз") == "none"  # "ашу" = to open
    assert bucket_of("Логотип жасап, футбол ойнаймын") == "none"
    assert bucket_of("Фу, сасық иіс") == "disgust"


def test_pool_uses_only_extreme_scores_and_hides_nothing_needed():
    texts = [f"Бұл қосымша туралы пікір нөмірі {i}" for i in range(300)]
    texts += [f"Қорқынышты жағдай болды, нөмір {i}" for i in range(50)]
    df = pd.DataFrame({"text": texts, "score": [1, 2, 3, 4, 5] * 70, "domain": "apps"})
    pool = make_pool(df, size=100, cue_share=0.5, seed=0)
    assert len(pool) == 100
    assert set(pool["score"]) <= {1, 2, 5}
    assert (pool["bucket"] == "fear").sum() >= 10
    assert pool["id"].is_unique
