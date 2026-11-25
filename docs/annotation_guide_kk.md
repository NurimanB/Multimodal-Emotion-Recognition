# Annotation guide: Kazakh text emotions

This guide is for labelling our own Kazakh set (and, if needed, extra Russian
text). The same 7 classes as MELD are used, so the set can be pooled with the
English and Russian corpora.

## What we label

Short real texts written by people: comments under Kazakh news and YouTube
videos, public Telegram channel posts and comments, and KazSAnDRA reviews
(only the text; its sentiment labels are not used). No translated or
generated text.

Before labelling, remove phone numbers, e-mails and personal names of private
people (replace with `[NAME]`). Keep public figures' names.

## Classes

Choose the emotion the **writer** expresses in this text, not the emotion the
text might cause in the reader.

| label | when to use | example |
|---|---|---|
| neutral | no clear emotion; facts, questions, info, ads | Ертең Алматыда жаңбыр жауады. |
| joy | happiness, pleasure, gratitude, love, pride, approval | Рахмет, керемет жаңалық! |
| surprise | unexpected, astonished, "really?!" (positive or negative) | Шынымен бе? Сенбеймін! |
| anger | irritation, outrage, insults, blame | Қашанға дейін алдай бересіңдер?! |
| sadness | grief, disappointment, regret, loneliness | Өте өкінішті, жаным ауырды. |
| disgust | revulsion, contempt, "yuck", moral disgust | Жиіркенішті, қалай ұялмайды. |
| fear | worry, anxiety, threat, "I'm scared that..." | Балаларға қауіпті ғой, қорқамын. |

Rules for hard cases:

1. One label per text. If two emotions are present, take the stronger or
   the last one. If you really cannot choose, mark `unclear`.
2. Sarcasm: label the real emotion (usually anger or disgust), not the
   surface words.
3. Mixed language (Kazakh + Russian) is fine; label it and keep it. Pure
   Russian texts go to the Russian file.
4. Emojis count: "😂😂" alone is joy, "😡" alone is anger.
5. Spam, ads, unreadable text, not Kazakh: mark `skip`.

## Process

1. Both of us label the same first 200 texts independently.
2. Compute Cohen's kappa. Discuss every disagreement and add an example to
   this guide if a rule was unclear. Target kappa >= 0.6.
3. Label the rest. Every text is labelled by both of us; disagreements are
   resolved together, and a text that stays `unclear` is dropped.
4. Split by source so the same Telegram thread or video is never in two
   splits: test 40%, dev 10%, train 50% (the set is mainly for evaluation).

Target size: 2,000-3,000 texts, at least 150 per class in test (disgust and
fear will need targeted searching, e.g. comments under news about accidents
or scandals).

## Making the pool

    python -m mer.data.kk_label_pool --kazsandra data/raw/kazsandra/full.csv --out data/own/kk_pool --size 3000

KazSAnDRA is gated on Hugging Face: log in, accept the terms on the dataset
page once, then download `full/full.csv`. The script keeps 1-2 and 5 star
reviews, removes duplicates, and takes half of the pool from texts with cue
words for the rare emotions (fear, disgust, surprise, sadness, anger) so they
are not missing. The annotators see only `id` and `text`; the star score and
cue bucket are in a separate file. Cue words only choose what to look at;
the label is always the annotator's decision.

KazEmoTTS transcripts are **not** used for text: the sentences come from
Wikipedia, news and books, and each sentence was given an emotion only to
tell the narrator how to read it (Abilbekov et al., 2024, section 3.2). The
label describes the voice, not the text. KazEmoTTS is for the audio stage.

## File format

`kk_annotated.csv`, UTF-8, columns:

`id, text, source, source_url, annotator_1, annotator_2, emotion, split`

`emotion` is the final agreed label. Build the pooled set with:

    python -m mer.data.build_text --config configs/data_text_v1.yaml \
        sources=[meld,emorynlp,dailydialog,goemotions,brighter_eng,carer,cedr,brighter_rus,kk_annotated] \
        kk_csv=data/own/kk_annotated.csv
