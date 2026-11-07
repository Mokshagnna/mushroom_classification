# Mushroom MLOps runbook

Run commands from the repository root in PowerShell.

## Routine checks

```powershell
uv run --python 3.12 --with-requirements requirements.txt python -m unittest discover -s tests -v
python src/mlops_pipeline/01_load_data.py
python src/mlops_pipeline/02_validate_data.py
python src/mlops_pipeline/03_preprocessing.py
python src/mlops_pipeline/04_validate_outputs.py
```

Train, evaluate, quality-gate, and register an eligible candidate:

```powershell
python src/mlops_pipeline/05_train_registry.py
python src/mlops_pipeline/06_model_evaluation.py
python src/mlops_pipeline/07_quality_gate.py
python src/mlops_pipeline/08_automate_lifecycle.py
```

`config.yaml` controls the dataset, paths, random seed, poisonous-recall
threshold, registry model name, and monitoring thresholds. `05_train_registry`
does not replace an incumbent with an inferior candidate; any replacement backs
up the prior model.

## Inference and service

```powershell
python -m src.mushroom_classification predict --input data/raw/mushroom.csv --output outputs/inference_predictions.csv
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

The service provides `/health`, `/ready`, `/metadata`, `/predict`, and
`/predict/batch`. Monitoring logs contain request IDs, predictions,
probabilities, latency, or bounded error identifiers, not feature rows.

## Diagnostics and data safety

```powershell
python src/mlops_pipeline/10_failure_injection.py
python src/mlops_pipeline/12_prediction_monitoring.py
python src/mlops_pipeline/12_api_performance_monitoring.py
python src/mlops_pipeline/12_monitor_trigger.py
```

Do not run corruption scenarios against the real raw data or model. The failure
runner uses temporary fixtures. Controlled drift generators write to
`data/monitoring/drift/`; their synthetic data and labels are not real
observations. Accuracy requires a separate validated dataset with actual
ground-truth labels.

## Deployment and MLflow

```powershell
docker compose config --quiet
docker compose up --build
python src/mlops_pipeline/11_prepare_deployment.py
python deployment/deploy_production.py
```

The deployment script validates preparation only unless `--apply` is passed.
Kubernetes application requires `kubectl` and explicit operator approval.
Registry candidates do not automatically become production versions.

The local MLflow SQLite database and run artifacts are preserved. For durable
CI tracking, configure `MLFLOW_TRACKING_URI` to an approved external server;
never write credentials into YAML or commit `.env` files.
