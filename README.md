# Multimodal emotion recognition

KBTU diploma project (6B06103 Mathematical and Computer Modeling): emotion recognition from text, audio and video, with Kazakh, Russian and English in scope. Work goes stage by stage; the first stage is text-only emotion recognition on MELD.

## Current stage: text branch on MELD

- Data: MELD text (train 9,989 / dev 1,109 / test 2,610 utterances, 7 emotions: neutral, joy, surprise, anger, sadness, disgust, fear).
- Models: RoBERTa-large and DeBERTa-v3-base with target text and previous dialogue turns; archived XLM-R-large results are also retained.
- Baseline: train-only TF-IDF with Logistic Regression.
- Metric: weighted F1 (also macro F1 and accuracy, per-class report and confusion matrix).
- Loss: class-weighted cross-entropy; the DeBERTa setup uses square-root inverse-frequency weights.

## Pooled text dataset (EN / RU / KK)

`python -m mer.data.build_text --config configs/data_text_v1.yaml` downloads the real text emotion corpora, maps every label to the 7 MELD emotions and writes `data/text_v1/{train,dev,test}.parquet` with `stats.md` (counts per source, language and class, and removed rows). On Colab use `notebooks/colab_build_text_data.ipynb`.

| source | lang | labels | notes |
|---|---|---|---|
| MELD | en | 7, same classes | never changed (benchmark) |
| EmoryNLP | en | 7 (feeling wheel) | Peaceful/Powerful dropped; copies of MELD lines removed |
| DailyDialog | en | 7, same classes | dialogues |
| GoEmotions | en | 27 + neutral | official Ekman grouping, single-label rows only |
| BRIGHTER (SemEval-2025 T11) | en, ru | 6 binary | no emotion = neutral, single-label only |
| CARER | en | 6 | labels from hashtags (`weak=True`), love = joy |
| CEDR | ru | 5 + no emotion | no emotion = neutral |
| own Kazakh set | kk | 7 | hand-labelled, see `docs/annotation_guide_kk.md` |

Machine-translated and LLM-generated corpora are not used. IEMOCAP transcripts can be added from a local CSV (`iemocap_csv=...`) once the licence is signed.

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

## DeBERTa-v3-base experiments

Open [notebooks/colab_deberta_meld.ipynb](notebooks/colab_deberta_meld.ipynb)
for the TF-IDF baseline, DeBERTa training, three-seed ensemble, evaluation
and inference. The T4 preset uses batch 4, accumulation 4, 256 tokens,
FP16, gradient checkpointing and best-dev checkpoint selection.

~~~bash
python -m pip install -r requirements-experiments.txt
python -m pip install -e .
python -m mer.text.train --config configs/text_deberta_v3_base.yaml
python scripts/run_deberta_multiseed.py
~~~

The equal mean of seeds 42, 123 and 2025 reaches test weighted F1 0.658646.
Combining it with RoBERTa reaches 0.668079 (macro F1 0.511280).
The benchmark had already been inspected in earlier iterations; within-run
checkpoint and ensemble selection use dev only.

Results follow the existing layout: one JSON per run, with all runs listed in
[results/model_comparison.csv](results/model_comparison.csv).
See [results/README.md](results/README.md) for the table, figures and limitations.

Training, inference, baselines and reporting share the utilities in
src/mer/evaluation. Existing model folders on Drive are located from recorded
run metadata. Raw probabilities, trainer logs and model weights stay outside
Git; selected metrics and figures are committed.

For a local source ZIP before pushing, run python scripts/build_colab_bundle.py
and upload dist/mer_experiments_colab.zip in the notebook setup cell.
