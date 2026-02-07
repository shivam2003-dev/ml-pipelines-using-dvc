# MLOps (DVC Pipeline)

This repo builds a small end-to-end text ML pipeline with DVC:
`data_ingestion` -> `data_preprocess` -> `feature_engineering` -> `model_training` -> `model_evaluation`.

## Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

Initialize Git + DVC (first time only):

```bash
git init
dvc init
```

## Run The Pipeline

Run all stages from `dvc.yaml`:

```bash
dvc repro
```

See the pipeline graph:

```bash
dvc dag
```

## Outputs

After `dvc repro` you should have:

- `data/raw/` (train/test split from ingestion)
- `data/preprocessed/` (cleaned text)
- `data/features/` (`train_bow.csv`, `test_bow.csv`)
- `models/model.joblib` (trained model)
- `reports/metrics.json` (DVC metrics)
- `reports/classification_report.txt`

View metrics:

```bash
dvc metrics show
cat reports/metrics.json
```

## (Optional) Recreate Stages With CLI

If you want to generate `dvc.yaml` via commands instead of editing:

```bash
dvc stage add -n data_ingestion \
  -d src/data_ingestion.py \
  -o data/raw \
  .venv/bin/python src/data_ingestion.py

dvc stage add -n data_preprocess \
  -d src/data_preprocessing.py \
  -d data/raw \
  -o data/preprocessed \
  .venv/bin/python src/data_preprocessing.py

dvc stage add -n feature_engineering \
  -d src/feature_engineering.py \
  -d data/preprocessed \
  -d params.yml \
  -o data/features \
  .venv/bin/python src/feature_engineering.py

dvc stage add -n model_training \
  -d src/model_building.py \
  -d data/features \
  -d params.yml \
  -o models/model.joblib \
  .venv/bin/python src/model_building.py

dvc stage add -n model_evaluation \
  -d src/model_evaluation.py \
  -d data/features \
  -d models/model.joblib \
  -M reports/metrics.json \
  -o reports/classification_report.txt \
  .venv/bin/python src/model_evaluation.py
```

## DVC Cheatsheet

Common commands:

```bash
dvc status
dvc repro
dvc repro -s model_training
dvc dag
dvc metrics show
dvc metrics diff
```

Useful Git workflow:

```bash
git status
git add dvc.yaml dvc.lock params.yml src
git commit -m "Add DVC pipeline"
```

Remote storage (optional):

```bash
dvc remote add -d myremote <REMOTE_URL>
dvc push
dvc pull
```

## Notes

- If you see `zsh: command not found: python`, use `python3` or `.venv/bin/python`.
- Parameters live in `params.yml` (e.g. BoW `max_features`, model hyperparameters).
