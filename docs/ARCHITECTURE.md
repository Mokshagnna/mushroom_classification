# Mushroom Classification Architecture

## Workflow

1. Load `data/raw/mushroom.csv` using the repository-root configuration.
2. Validate the 23-column schema, target labels, empty values, duplicates, class distribution, constant features, and the `?` unknown marker in `stalk-root`.
3. Split features and target using a stratified 80/20 split and a fixed seed.
4. Compare Logistic Regression, Decision Tree, Random Forest, and K-Nearest Neighbors using stratified cross-validation on the training partition. Each fold fits its own imputer and one-hot encoder, preventing preprocessing leakage.
5. Select the candidate with the best mean poisonous-class recall; use variability as a tie-breaker.
6. Fit the selected preprocessing/model pipeline on the training partition and evaluate once on the held-out test partition.
7. Write validation and evaluation reports, apply the configured poisonous-recall quality gate, and compare a candidate with the incumbent on the same deterministic test split. A passing but inferior candidate does not replace the incumbent; any replacement is timestamp-backed-up.
8. Optionally log and register the passing candidate with MLflow.
9. Load the same serialized pipeline for CSV and API inference. Unknown feature categories are ignored by the encoder; missing `stalk-root` values are imputed.
10. Serve validated single or batch feature records through FastAPI and record prediction class, poisonous probability, latency, and failures without persisting feature values. Accuracy is not calculated without labels.
11. Run stage-specific data/prediction drift monitors and labeled performance checks. Controlled synthetic drift data is segregated under `data/monitoring/drift/` and always identified as synthetic.

The class mapping is `e -> 0` (edible) and `p -> 1` (poisonous). Confusion-matrix labels and the quality gate use that same ordering. Poisonous recall is emphasized because a poisonous-as-edible error is especially consequential, but no metric makes this a safe foraging tool.

## Churn reference comparison

The separate Churn Prediction reference was inspected read-only. It is a flat 145-file archive; `PROJECT_STRUCTURE.txt` describes its original layout and `FILE_PLACEMENT_MAP.csv` maps each archived file to that layout. Its implementation is strongly coupled to Telco customer records and is not copied into this project.

| Component | Churn reference | Mushroom project | Migration decision |
|---|---|---|---|
| Data contract | `src/mlops_pipeline/02_validate_data.py`, `src/mlops_pipeline/validation.py`; customer identifiers and Telco fields | `src/mushroom_classification/validation.py`; 23-column categorical schema and `?` unknown marker | Keep the Mushroom schema; do not introduce customer IDs or churn labels. |
| Preprocessing | `src/mlops_pipeline/03_preprocessing.py`; numeric scaling, one-hot encoding, fixed customer-ID test set | `src/mushroom_classification/models.py`; fold-local imputation and one-hot encoding inside the estimator pipeline | Adapt the no-leakage pipeline pattern; retain mushroom-specific preprocessing. |
| Training and quality | `src/mlops_pipeline/churn_prediction_pipeline.py`, `src/mlops_pipeline/07_quality_gate.py`; churn-oriented model and recall gate | `src/mushroom_classification/pipeline.py`; cross-validated model comparison and poisonous-recall gate | Use a single mushroom workflow and threshold; no churn-specific training logic. |
| Registry and lifecycle | `src/mlops_pipeline/05_train_registry.py`, `src/mlops_pipeline/08_automate_lifecycle.py`; registry and lifecycle operations | `src/model_registry.py`, `src/mushroom_classification/pipeline.py`; configured MLflow store and candidate registration | Keep tracking configurable and preserve existing registry history; do not automate production promotion. |
| Inference/API | `api/main.py`, `api/preprocessing.py`; FastAPI customer endpoints and churn probability | `api/main.py`, `api/schemas.py`, and `src/mushroom_classification/inference.py`; 22 mushroom features and the saved full pipeline | Adapted FastAPI lifecycle, validation-error response, model readiness, and prediction endpoint patterns. Replaced all customer fields and churn outputs. |
| Testing and CI | `tests/test_api.py`, `tests/test_performance.py`, `.github/workflows/lab8_ci.yml` | Mushroom unit tests and mushroom-specific GitHub workflows | Adapted API request and health testing patterns; added mushroom-specific API, monitoring, runner, and isolated failure tests. |
| Monitoring/retraining | `monitoring_logger.py`, `12_prediction_monitoring.py`; churn prediction distribution and probability summaries | `src/mushroom_classification/monitoring.py`; mushroom prediction class, poisonous probability, latency, and failures | Adapted append-only event logging and summary metrics. No feature values are stored; accuracy is omitted without ground-truth labels. Churn drift/retraining logic is excluded. |
| Deployment | `Dockerfile`, `deployment/`, `k8s/`; production deployment assets | Root/deployment Dockerfiles, `docker-compose.yml`, `k8s/`, and guarded deployment script | Adapted slim container, health probes, and manifests for the saved mushroom pipeline. Deployment execution requires explicit operator approval and was not claimed as tested. |

No reference files were extracted or copied. Relevant architectural patterns were implemented in the Mushroom package using its existing data, configuration, registry, and CI context. The separate reference archive remains unchanged.

## Component assessment

| Component | Status | Notes |
|---|---|---|
| Data validation | Implemented | Mushroom-specific schema, target, missing/unknown marker, duplicates, and constants. |
| Preprocessing | Implemented | Imputation and one-hot encoding are part of the fitted sklearn pipeline. |
| Train/test split | Implemented | Stratified, seeded, and separated before model selection. |
| Model comparison | Implemented | CV comparison across four existing model families; selection optimizes poisonous recall. |
| Evaluation and quality gate | Implemented | Holdout metrics, class-oriented confusion matrix, and poisonous-recall threshold. |
| Experiment tracking / registry | Implemented, configurable | Uses the existing Mushroom registry helper and MLflow tracking URI. |
| Model versioning | Implemented | New model versions receive a `candidate` alias; existing registry history is preserved. |
| Promotion / deployment | Guarded, not automatic | Candidates are quality-gated; deployment metadata and manifests exist. No automatic production promotion or deployment occurs. |
| Reproducibility | Implemented | Fixed split, CV, and model random states; reports include a dataset SHA-256. |
| API / Docker | Implemented for local demonstration | FastAPI serves the saved mushroom pipeline with strict 22-feature validation. Compose builds and runs a local container. |
| Kubernetes / production deployment | Configuration prepared, runtime unverified | Manifests and guarded `kubectl --apply` command exist; no Kubernetes cluster was available for deployment verification. |
| Prediction monitoring | Implemented for operational statistics | Distribution, probabilities, latency, and failures are tracked. Accuracy is unavailable without ground-truth labels. |
| Drift analysis | Implemented as diagnostics | Feature and prediction distribution changes are reportable. Concept shift requires labeled comparisons; synthetic drift labels are flagged and not treated as ground truth. |
| Safe retraining | Explicit command only | Candidate trains to a temporary artifact, passes the configured gate, and must match or exceed incumbent poisonous recall on the same holdout before replacement. |
| Failure handling | Implemented in tests | Invalid schemas and inference feature mismatches fail explicitly; CI exercises regression tests. |

## Directory and history decisions

- Keep the supplied raw dataset, its existing metadata, prior model artifacts, experiment databases, MLflow files, reports, and notebooks in place.
- Keep `src/mushroom_classification/` as the canonical workflow. `src/mlops_pipeline/` provides numbered, mushroom-specific orchestration/monitoring entry points.
- Retain the churn notebook, churn-specific pipeline, and churn lab reports in place as inactive history. They are not invoked by the new mushroom CI workflows.
- Treat the pre-existing notebook/model reports as historical: their target mapping and confusion-matrix labels were inconsistent. Newly generated pipeline reports consistently use poisonous as class `1`.
- API, Docker, and Kubernetes configuration are demonstrations. Docker Compose syntax is checked; an actual container build and Kubernetes apply remain unverified.
