# Screenshots to capture (name files exactly like this)

| File | What to show |
|---|---|
| mlflow_runs.png | MLflow experiment table with all runs + metrics columns |
| mlflow_compare.png | Compare view of the 3 model runs |
| mlflow_artifacts.png | Artifacts of one run (plots, model, signature) |
| ci_success.png | GitHub Actions run, all 4 jobs green |
| ci_artifacts.png | Artifacts list at the bottom of the run page |
| ci_failure.png | A run that FAILED (e.g. push a broken test, screenshot, then revert) |
| docker_build.png | `docker build` finishing successfully |
| docker_run_predict.png | `docker run` + curl /predict response |
| swagger_ui.png | http://localhost:8000/docs |
| k8s_resources.png | `kubectl get deploy,pods,svc,ingress,hpa` |
| k8s_predict.png | curl through the LoadBalancer/Ingress |
| k8s_logs.png | `kubectl logs deploy/heart-disease-api` showing JSON request logs |
| prometheus_targets.png | Prometheus -> Status -> Targets (UP) |
| grafana_dashboard.png | Grafana dashboard populated after generate_traffic.py |
