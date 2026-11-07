# Mushroom project Git workflow

1. Create a feature branch for each lab or change.
2. Keep the supplied raw mushroom dataset immutable; place controlled drift
   experiments under `data/monitoring/drift/`.
3. Run unit tests, the staged workflow, and isolated failure checks as needed.
4. Review the diff and ensure credentials, MLflow history, local models, and
   generated outputs are not staged.
5. Merge only after the applicable GitHub Actions checks pass.

The `.gitignore` excludes local MLflow databases, model files, reports, logs,
monitoring datasets, and processed data. Commit source, configuration,
documentation, workflows, and approved input data only. Do not treat generated
drift datasets as real mushroom observations or food-safety evidence.
