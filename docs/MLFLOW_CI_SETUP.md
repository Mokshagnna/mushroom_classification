# MLflow and CI setup

The Mushroom workflow continues to use `MushroomModelRegistry` and the
`mushroom-classification` experiment configured in `config.yaml`. Registry
versions are recorded as candidates; lifecycle automation does not silently
move a version to production.

The existing SQLite database at `artifacts/mlflow.db` and its history are
preserved. To use a persistent server, set `MLFLOW_TRACKING_URI` in PowerShell
or configure it as a GitHub Actions secret:

```powershell
$env:MLFLOW_TRACKING_URI = "https://your-approved-mlflow-server"
uv run --python 3.12 --with-requirements requirements.txt python -m src.mushroom_classification train
```

GitHub-hosted runners are ephemeral. Without an external tracking server,
CI-run experiment history does not persist after the job. Do not commit access
tokens, `.env`, tracking databases, or artifact stores.
