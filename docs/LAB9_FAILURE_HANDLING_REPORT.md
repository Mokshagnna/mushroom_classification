# Mushroom failure handling

The isolated failure-injection runner is:

```powershell
uv run --python 3.12 --with-requirements requirements.txt python -m src.mlops_pipeline.10_failure_injection
```

It exercises missing data, schema mismatch, unsupported categories, malformed
CSV, missing/corrupt model artifacts, healthy API startup and prediction, and
invalid API requests. Temporary paths are used for corrupt input and models;
the shipped raw dataset and configured model are not moved or overwritten.

The report is written to `reports/failure_report.json` and a run summary to
`artifacts/failure_simulation_report.json`. A failure to observe the expected
error exits non-zero. These checks are also covered by the unit/API tests in CI.
