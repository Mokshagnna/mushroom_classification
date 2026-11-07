# Lab 11 — Deployment verification and debugging notes

The Compose file parses successfully and points to the root `Dockerfile`.
The image build was attempted, but Docker returned an unavailable
`dockerDesktopLinuxEngine` named pipe error. This indicates the local Docker
engine was not running/reachable; it is not evidence that the image built or
the container started.

The Kubernetes manifests are structurally valid YAML and define a deployment
with readiness/liveness HTTP probes plus a ClusterIP service. No cluster or
`kubectl apply` run was available, so Kubernetes runtime status is unverified.

Before local deployment, start Docker Desktop and run:

```powershell
docker info
docker compose config --quiet
docker compose up --build
docker compose ps
Invoke-RestMethod http://127.0.0.1:8000/ready
```

Before Kubernetes deployment, build/publish the image for the target cluster,
verify the mushroom model is in the image, and review the quality gate. Only
then use `python deployment/deploy_production.py --apply` with explicit
authorization.
