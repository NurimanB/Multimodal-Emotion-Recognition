# Multimodal emotion recognition

KBTU diploma project (6B06103 Mathematical and Computer Modeling): emotion recognition from text, audio and video, with Kazakh, Russian and English in scope.

The work goes stage by stage. The first stage is text-only emotion recognition on the MELD dataset.

## Plan

```
configs/       run settings (YAML)
notebooks/     Colab notebooks
scripts/       data download and inspection
src/mer/       Python package
tests/         unit tests
```

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

## Data: MELD

```bash
python scripts/download_meld.py --data_dir data/meld
```

Downloads the MELD text CSVs (train 9,989 / dev 1,109 / test 2,610 utterances) and prints the class balance. Seven emotions: neutral, joy, surprise, anger, sadness, disgust, fear. Neutral is 47% of the training set.

Each utterance can be given the previous N turns of its dialogue (with speaker names) as context, see `add_context` in `src/mer/data/meld.py`.

```bash
pytest
```

## Training (text)

```bash
python -m mer.text.train --config configs/smoke.yaml
```

Any `key=value` after the config overrides that setting, e.g. `n_context=5 lr=2e-5`. Metric: weighted F1 (also macro F1 and accuracy).
