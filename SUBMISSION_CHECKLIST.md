# Submission checklist (mapped to the marking scheme)

Run everything yourself, look at the output, and write the report in your own words - the assignment's integrity
note says similar submissions are penalised, so change names, model choices, features and wording to reflect
your own decisions.

| # | Task (marks) | Where | Done when |
|---|---|---|---|
| 1 | Data + EDA (5) | `scripts/download_data.py`, `src/eda.py`, notebook 01, `reports/figures/` | `data/raw` + `data/processed` committed; 6 figures; observations written |
| 2 | Features + models (8) | `src/features.py`, `src/modeling.py` | >= 2 models, nested CV, accuracy/precision/recall/ROC-AUC reported |
| 3 | MLflow (5) | `src/train.py` | `mlruns` screenshots: params, metrics, plots, models for all runs |
| 4 | Packaging (7) | `models/`, `requirements*.txt` | clean-venv run works; pipeline saved as one artefact |
| 5 | CI/CD + tests (8) | `tests/`, `.github/workflows/ci.yml` | green run screenshot **and** a red run screenshot |
| 6 | Docker (5) | `Dockerfile` | build + run + `/predict` screenshots |
| 7 | K8s deployment (7) | `k8s/` | pods/svc/ingress screenshots + curl via LoadBalancer |
| 8 | Monitoring (3) | `/metrics`, `monitoring/`, `k8s/monitoring.yaml` | Grafana dashboard with data + logs screenshot |
| 9 | Report (2) | `docs/REPORT.md` -> PDF (about 10 pages) | all `[[FILL]]` placeholders replaced |
| - | Video | `VIDEO_SCRIPT.md` | 4-5 min recording, link in report/README |

## Order of work
1. `pip install -r requirements.txt`, `python scripts/download_data.py`, `python -m src.eda`, `python -m src.train`
2. `pytest`, `flake8 src tests scripts` - fix anything that fails on your machine.
3. Commit **including** `data/`, `models/model.joblib`, `models/model_info.json`, `reports/` (Dockerfile needs the model).
4. Push to GitHub -> watch Actions -> screenshots (also create one failing run).
5. Docker build/run/smoke test -> screenshots.
6. Minikube/Docker Desktop deploy -> screenshots; monitoring + `generate_traffic.py` -> screenshots.
7. Fill the report, export PDF, record video, submit repo link + report + video.
