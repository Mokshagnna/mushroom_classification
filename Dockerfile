FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

COPY config.yaml ./
COPY src/__init__.py ./src/__init__.py
COPY src/mushroom_classification/ ./src/mushroom_classification/
COPY api/ ./api/
COPY models/mushroom_classifier.joblib ./models/mushroom_classifier.joblib

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
