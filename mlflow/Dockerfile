FROM python:3.11-slim

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout 120 --retries 5 mlflow

EXPOSE 5000

CMD ["mlflow", "server", \
     "--host", "0.0.0.0", "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlflow-data/mlflow.db", \
     "--serve-artifacts", \
     "--artifacts-destination", "/mlflow-data/mlruns", \
     "--default-artifact-root", "mlflow-artifacts:/"]