# 4-5 minute demo video - suggested script (screen-record with voice-over)

1. **(0:00)** Title + problem in one sentence; show repo tree in VS Code / GitHub.
2. **(0:30)** Terminal: `pip install -r requirements.txt`, `python scripts/download_data.py`.
3. **(0:50)** Show EDA figures (class balance, heatmap) - one insight each.
4. **(1:15)** `python -m src.train` (speed up / cut), open **MLflow UI**: runs, compare, artifacts.
5. **(2:00)** `pytest` green; GitHub Actions page: 4 green jobs, artifacts. Mention a failing run fails the pipeline.
6. **(2:30)** `docker build` + `docker run`; call `/predict` via Swagger or curl.
7. **(3:00)** `kubectl apply -f k8s/...`; `kubectl get pods,svc`; call the API through the LoadBalancer/Ingress.
8. **(3:40)** Run `generate_traffic.py`; show Prometheus targets and the Grafana dashboard moving; show `kubectl logs`.
9. **(4:20)** Wrap-up: architecture diagram, what you would improve next.
