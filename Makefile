.PHONY: install data eda train test lint serve docker-build docker-run smoke mlflow-ui monitoring k8s-up all
PY ?= python

install:
	$(PY) -m pip install -r requirements.txt
data:
	$(PY) scripts/download_data.py
eda:
	$(PY) -m src.eda
train:
	$(PY) -m src.train
test:
	$(PY) -m pytest --cov=src --cov-report=term-missing
lint:
	$(PY) -m flake8 src tests scripts
serve:
	$(PY) -m uvicorn src.api:app --reload --port 8000
docker-build:
	docker build -t heart-disease-api:1.0.0 .
docker-run:
	docker run --rm -p 8000:8000 heart-disease-api:1.0.0
smoke:
	bash scripts/smoke_test.sh http://localhost:8000
mlflow-ui:
	mlflow ui --backend-store-uri ./mlruns --port 5000
monitoring:
	docker compose -f monitoring/docker-compose.yml up --build
k8s-up:
	kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml -f k8s/ingress.yaml -f k8s/hpa.yaml
all: data eda train test lint
