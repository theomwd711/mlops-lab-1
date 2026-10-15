FROM python:3.11-slim

WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout 120 --retries 5 streamlit requests

COPY app.py .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]