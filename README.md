# Mushroom Classification MLOps

An academic classification project for the UCI mushroom dataset. It validates categorical data, compares models with stratified cross-validation, fits preprocessing only on training folds, evaluates a held-out test split, and optionally tracks/registers the selected model with MLflow.

> **Food-safety warning:** This model is for academic demonstration only. Never use its predictions to decide whether a mushroom is safe to eat.

## Dataset

The supplied CSV is `data/raw/mushroom.csv` (8,124 rows, 22 categorical features, and target `class`). Labels are `e` (edible) and `p` (poisonous); the training workflow encodes poisonous as positive class `1`. The `stalk-root` feature uses `?` for unknown values. Validation reports this sentinel, and the fitted pipeline imputes missing values and ignores previously unseen categories.

## Architecture

```text
data/raw/mushroom.csv
          |
          v
schema validation -> stratified holdout split
          |                   |
          |                   v
          |      CV model comparison (preprocessing inside each fold)
          |                   |
          |                   v
          +--------> selected sklearn Pipeline
                              |
                held-out evaluation + poisonous-recall gate
                              |
          reports/ + models/mushroom_classifier.joblib
                              |
                optional MLflow tracking/registry
                              |
                    batch CSV inference
```

See [the architecture and MLOps assessment](docs/ARCHITECTURE.md) for workflow details and retained legacy components.

## Project structure

```text
.
├── .dockerignore               # Exclude local datasets, artifacts, and secrets
├── .github/workflows/          # Unit, failure, staged runner, and monitoring CI
├── api/                        # FastAPI, schemas, model loading, preprocessing, monitoring
├── artifacts/                  # Existing MLflow databases/history (preserved)
├── Dockerfile                  # Python 3.12 API image
├── data/
│   ├── raw/mushroom.csv        # Original dataset (never modified by failure tests)
│   ├── processed/              # Generated splits/metadata
│   └── monitoring/drift/       # Segregated generated drift experiments
├── docs/                       # Architecture and retained lab documentation
├── deployment/
│   ├── Dockerfile              # Repository-root build context
│   └── deploy_production.py    # Explicit, gate-checked kubectl apply
├── docker-compose.yml          # Local API deployment
├── k8s/                        # Deployment and service manifests
├── models/                     # Existing models plus new fitted end-to-end pipeline
├── notebooks/
│   └── mushroom_classification.ipynb
├── outputs/                    # Existing experiment reports and inference outputs
├── pipelines/
│   └── run_mushroom_pipeline.py
├── reports/                    # Generated validation/evaluation reports
├── requirements-api.txt        # Minimal API container dependencies
├── src/mlops_pipeline/         # Numbered stages and monitoring entry points
├── src/mushroom_classification/
│   ├── config.py
│   ├── logging_config.py
│   ├── monitoring.py
│   ├── validation.py
│   ├── models.py
│   ├── evaluation.py
│   ├── pipeline.py
│   └── inference.py
└── tests/
    ├── test_mushroom_classification.py
    ├── test_api.py
    └── test_monitoring.py
```

The existing churn notebook and unrelated historical root wrappers are retained
as inactive history. Numbered `src/mlops_pipeline/` scripts have been rewritten
as mushroom stage entry points that reuse the canonical package.

Existing models, notebook work, reports, and MLflow runs are preserved as historical assets. Some pre-existing notebook summaries used an edible-positive mapping while labeling the confusion matrix in the opposite order; the mushroom pipeline's newly generated reports use poisonous as class `1` and are the authoritative results for this workflow.

## Setup

Use Python 3.12 for the verified workflow and install the project dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Train, evaluate, and register

From the project root:

```powershell
python -m src.mushroom_classification train
```

The command validates the dataset, uses a stratified train/test split, compares Logistic Regression, Decision Tree, Random Forest, and K-Nearest Neighbors using training-only cross-validation, and selects by poisonous recall. It reports accuracy, macro precision/recall/F1, class-specific metrics, ROC-AUC, a correctly ordered confusion matrix, and the quality-gate result.

The default quality gate requires poisonous recall of at least `0.95`. This is a project demonstration threshold, not a safety certification or guarantee. Training saves the complete preprocessing/model pipeline to `models/mushroom_classifier.joblib`, along with validation and evaluation JSON under `reports/`. When that configured artifact already exists, the candidate is compared using the same deterministic held-out split; an inferior candidate is not installed. A replacement is timestamp-backed-up.

The MLflow registry is enabled in `config.yaml`. To train without registry side effects, set `mlflow.enabled: false`. A remote tracking server can be selected with `MLFLOW_TRACKING_URI`; never store credentials in the config file.

## Batch inference

Provide a CSV with all 22 feature columns (the `class` target may be omitted):

```powershell
python -m src.mushroom_classification predict --input data/raw/mushroom.csv --output outputs/inference_predictions.csv
```

The output includes predicted class labels and edible/poisonous probabilities. Predictions are not reliable food-safety advice.

## Tests and CI

```powershell
python -m unittest discover -s tests -v
```

GitHub Actions runs the unit tests and the training/evaluation workflow on relevant code, configuration, dependency, and dataset changes. Generated model files, reports, MLflow data, and processed data remain local/CI artifacts according to `.gitignore`.

## API and local container

The FastAPI application loads the configured saved sklearn pipeline
(`models/mushroom_classifier.joblib`) at startup and uses the same feature
normalization and preprocessing path as batch inference. Requests use the
dataset's 22 hyphenated feature names and categorical values. `class` and
customer-related fields are not accepted.

Run locally after installing `requirements.txt`:

```powershell
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

`GET /health`, `GET /ready`, `GET /metadata`, `POST /predict`, and
`POST /predict/batch` are available. Prediction responses include the
established `0 = edible`, `1 = poisonous` code and class probabilities. The
append-only prediction log is stored at
`logs/inference/mushroom_predictions.jsonl`; it records operational summaries
without feature values. Monitoring reports prediction distribution and latency,
not accuracy, because no ground-truth labels are supplied at inference time.
The API and local Docker setup are demonstrations only, not a production or
food-safety system.

Build and start with Docker Compose:

```powershell
docker compose up --build
```

The container uses Python 3.12 and the API-only dependency manifest, exposes
port 8000, mounts the model directory read-only, and persists monitoring events
under `logs/`.

## Stages and monitoring

Stage scripts are in `src/mlops_pipeline/`; the integrated runner is:

```powershell
uv run --python 3.12 --with-requirements requirements.txt python -m src.mlops_pipeline.mushroom_classification_pipeline
```

Stage 12 commands provide feature/prediction drift diagnostics and labeled
performance evaluation. Controlled drift generators write only under
`data/monitoring/drift/`; generated labels are explicitly synthetic and are
never used as real ground truth. Prediction statistics do not calculate
accuracy without actual labels. See [the runbook](docs/RUNBOOK.md) and
[end-to-end guide](docs/LAB13_END_TO_END.md).
