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

## Latest text experiments

The final four-checkpoint ensemble combines RoBERTa with the equal probability
mean of DeBERTa seeds 42, 123 and 2025. On the official MELD test split
(n = 2,610), it reaches weighted F1 **0.668079**, macro F1 **0.511280**
and accuracy **0.677395**.

| Run | Dev weighted F1 | Test weighted F1 | Test macro F1 |
|---|---:|---:|---:|
| TF-IDF + Logistic Regression | 0.418959 | 0.416621 | 0.248710 |
| RoBERTa recheck | 0.637366 | 0.613232 | 0.476107 |
| DeBERTa seed 42 | 0.632412 | 0.640631 | 0.469680 |
| DeBERTa seed 123 | 0.631273 | 0.642999 | 0.493418 |
| DeBERTa seed 2025 | 0.620923 | 0.646693 | 0.492233 |
| Equal three-seed probability mean | 0.639328 | 0.658646 | 0.506526 |
| Frozen RoBERTa + DeBERTa mean | 0.668962 | 0.668079 | 0.511280 |

All within-v3 checkpoint and blend choices used dev. This is an exploratory
continuation after earlier test inspection. Fear/disgust and external-domain
generalization remain limitations.

See [results](results/README.md), [experiment protocol](docs/experiment_protocol.md)
and [final run analysis](docs/final_results.md).

## Baseline, evaluation and inference

~~~bash
python -m pip install -r requirements-experiments.txt
python -m pip install -e .
python -m mer.evaluation.audit --data-dir data/meld --near-duplicates
python -m mer.evaluation.baseline --data-dir data/meld --output-dir outputs/experiments/tfidf
python -m mer.text.train --config configs/text_deberta_v3_base.yaml
python -m mer.evaluation.report --run-dir outputs/experiments/deberta_s42
python -m mer.evaluation.inference --checkpoint outputs/experiments/deberta_s42/best_model --config outputs/experiments/deberta_s42/config.json --text "I am so happy to see you!" --speaker Rachel
~~~

For Colab, use [the full experiment notebook](notebooks/colab_experiments.ipynb),
[additional seeds](notebooks/colab_extra_seeds.ipynb), or
[frozen ensemble inference](notebooks/colab_ensemble_demo.ipynb).
Build a source ZIP before committing with:

~~~bash
python scripts/build_colab_bundle.py
~~~

Upload dist/mer_experiments_colab.zip in the notebook setup cell.
The package contains current source; data and large weights are excluded.

## Reproducibility

Configs, sample IDs, split fingerprints, trainer state, probabilities and frozen
selection hashes identify each experiment. Raw run archives in
results/experiments/evidence are retained byte-for-byte.
Model weights remain on Drive. Existing run folders are found by their metadata;
new experiments use MyDrive/diploma-mer/experiments.

The T4 preset uses batch 4, accumulation 4, max length 256, mixed precision,
gradient checkpointing and best-dev checkpoint selection.
Full optimizer/RNG resume requires save_only_model=false and persistent
checkpoints. The default model-only local checkpoints do not provide exact
training resume after a VM is deleted.

Archived result attribution is documented in
[legacy_results_provenance.json](results/legacy_results_provenance.json).
