# Mushroom CI workflow

`.github/workflows/lab8_ci.yml` runs the Mushroom Classification training,
quality gate, tests, and artifact upload on relevant source/data/configuration
changes. Python 3.12 is used to match the supported local workflow and serving
image.

CI trains before running API tests because model binaries are local generated
artifacts and are not assumed to be present in a clean checkout. Each hosted
runner has an ephemeral workspace; MLflow state is not a persistent registry
unless `MLFLOW_TRACKING_URI` is configured to a separately managed tracking
service.

Run the same unit suite locally:

```powershell
uv run --python 3.12 --with-requirements requirements.txt python -m unittest discover -s tests -v
```

The hosted workflow's report and model uploads are diagnostic build artifacts,
not deployments or food-safety certifications.
