# Lab 11 — Mushroom API containerization

The root `Dockerfile` and `deployment/Dockerfile` use Python 3.12, install the
API-only manifest, copy the FastAPI package and configured full mushroom
pipeline, and expose port 8000. `.dockerignore` excludes datasets, MLflow
history, logs, reports, notebooks, tests, credentials, and unrelated model
artifacts while allowing the configured model file.

`docker-compose.yml` binds the API to localhost, mounts `models/` read-only,
persists logs, and checks `/health`. The syntax was validated with
`docker compose config --quiet`. The Docker image was not built in the
verification environment because the Docker Desktop Linux engine was
unavailable; container startup is therefore not reported as tested.

Build and run when the engine is available:

```powershell
docker compose up --build
Invoke-RestMethod http://127.0.0.1:8000/health
```

This local API is an academic demonstration and must not be used for foraging
or food-safety decisions.
