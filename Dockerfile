FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
ENV PYTHONPATH=/app/src MODEL_PATH=/app/artifacts/model.joblib METRICS_PATH=/app/artifacts/metrics.json
COPY artifacts ./artifacts
EXPOSE 8000
CMD ["uvicorn", "acuicultura.api:app", "--host", "0.0.0.0", "--port", "8000"]
