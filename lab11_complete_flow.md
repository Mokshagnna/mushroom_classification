# Lab 11 flow — Mushroom API deployment

1. Train and evaluate the mushroom pipeline.
2. Verify the configured quality gate and incumbent comparison.
3. Run `python src/mlops_pipeline/11_prepare_deployment.py`.
4. Validate Docker configuration with `docker compose config --quiet`.
5. If the Docker engine is available, run `docker compose up --build`.
6. Check `/health` and `/ready`.
7. Review Kubernetes manifests before any approved `--apply`.

The API uses the 22 mushroom features and the fitted pipeline. This
demonstration is not suitable for real-world foraging or food-safety decisions.
