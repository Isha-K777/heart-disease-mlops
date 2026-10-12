FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/models/model.joblib \
    MODEL_INFO_PATH=/app/models/model_info.json

WORKDIR /app

# Install deps first for better layer caching
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY src/ ./src/
COPY models/model.joblib models/model_info.json ./models/

# Run as non-root
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
