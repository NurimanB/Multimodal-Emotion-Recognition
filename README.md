# Multimodal emotion recognition

KBTU diploma project (6B06103 Mathematical and Computer Modeling): emotion recognition from text, audio and video, with Kazakh, Russian and English in scope. Work goes stage by stage; the first stage is text-only emotion recognition on MELD.

## Current stage: text branch on MELD

- Data: MELD text (train 9,989 / dev 1,109 / test 2,610 utterances, 7 emotions: neutral, joy, surprise, anger, sadness, disgust, fear).
- Main model: `xlm-roberta-large` fine-tuned on the target utterance plus the previous turns of the dialogue (with speaker names).
- Reference: `roberta-large` with the same setup.
- Metric: weighted F1 (also macro F1 and accuracy, per-class report and confusion matrix).
- Loss uses inverse-frequency class weights because neutral is 47% of training data.

## Layout

```
configs/            run settings (YAML), one file per experiment
notebooks/          Colab notebook: clone, install, download data, train
scripts/            data download and inspection
src/mer/data/       dataset loading (meld.py)
src/mer/text/       text model training (train.py)
tests/              unit tests (pytest)
```

Data, checkpoints and outputs are not committed (see `.gitignore`).

## Run on Colab

Open `notebooks/colab_text_meld.ipynb` in Colab (File → Open notebook → GitHub), pick a GPU runtime, and run the cells. Results are written to `MyDrive/diploma-mer/outputs/<run>/results.json`.

## Run locally

```bash
pip install -r requirements.txt
pip install -e .
python scripts/download_meld.py --data_dir data/meld
python -m mer.text.train --config configs/smoke.yaml
python -m mer.text.train --config configs/text_xlmr_large.yaml n_context=5 lr=2e-5
pytest
```

Any `key=value` after the config overrides that setting (see `DEFAULTS` in `src/mer/text/train.py`).

## Working together

- `main` should always run. Do new work on a branch (`text-context-window`, `audio-baseline`, ...) and merge with a pull request.
- One config file per experiment that goes into the thesis, so every number can be reproduced.
- Log every training run in the Notion Experiments table (run name, config, dev/test weighted F1).
