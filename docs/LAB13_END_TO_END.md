# Mushroom end-to-end MLOps workflow

The project uses the existing mushroom dataset and canonical package. Labels are
`e = edible`, `p = poisonous`, encoded as `0` and `1` respectively. The 22
categorical feature schema and train-fitted imputation/one-hot pipeline are
shared by training, batch inference, and the HTTP API.

## Run the stages

From PowerShell at the project root:

```powershell
uv run --python 3.12 --with-requirements requirements.txt python -m src.mlops_pipeline.mushroom_classification_pipeline
```

The runner executes loading, validation, split/preprocessing, output validation,
candidate training/MLflow registry, holdout evaluation, quality gate, and
lifecycle reporting. It writes `logs/mushroom_classification_pipeline_run_log.json`.
To isolate outputs, pass a YAML config containing temporary absolute paths:

```powershell
python src/mlops_pipeline/01_load_data.py --config C:\temp\mushroom-config.yaml
```

`05_train_registry.py` uses the canonical `run_pipeline`, preserving model
backups and MLflow tracking. Candidate versions are registered without
automatic production promotion. The incumbent is replaced only if the
candidate passes the poisonous-recall gate and is at least as good on the same
seeded holdout.

## API

```powershell
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Use `/health`, `/ready`, `/metadata`, `/predict`, and `/predict/batch`. The API
accepts the dataset's hyphenated feature names; `class` and unrelated fields
are rejected. Responses return class code, label, and both probabilities.

## Monitoring and controlled drift

```powershell
python src/mlops_pipeline/12_prediction_monitoring.py
python src/mlops_pipeline/12_api_performance_monitoring.py
python src/mlops_pipeline/12_create_evolved_dataset.py --seed 2026 --rows 1000
python src/mlops_pipeline/12_data_drift.py --input data/monitoring/drift/evolved_data.csv
python src/mlops_pipeline/12_create_concept_drift_dataset.py --seed 2026 --rows 1000
python src/mlops_pipeline/12_concept_drift.py --baseline data/raw/mushroom.csv --current data/monitoring/drift/evolved_labeled_data.csv
```

Feature drift and prediction drift are distribution diagnostics, not accuracy
or evidence of concept drift. Concept-shift comparison requires labels; labels
in the controlled drift file are deliberately perturbed and flagged as
synthetic, never trusted ground truth. Real performance monitoring requires an
explicit labeled dataset:

```powershell
python src/mlops_pipeline/12_performance_monitoring.py --input data/raw/mushroom.csv
python src/mlops_pipeline/12_retrain_model.py
```

Retraining uses a temporary candidate artifact and preserves the incumbent
unless the candidate passes the configured gate and does not reduce poisonous
recall. The retraining command does not promote an MLflow version.

## Deployment

`docker compose config --quiet` validates Compose configuration. Build/run with
`docker compose up --build` when a Docker engine is available. Kubernetes
manifests are under `k8s/`; inspect prepared metadata and quality-gate reports
before an explicitly approved `python deployment/deploy_production.py --apply`.
Neither a manifest nor a model score is a food-safety certification.
