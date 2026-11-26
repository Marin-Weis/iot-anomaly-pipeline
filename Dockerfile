FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src

RUN useradd --system --create-home appuser && mkdir -p /app/data /app/models && chown -R appuser:appuser /app
USER appuser

EXPOSE 8083

HEALTHCHECK --interval=15s --timeout=3s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8083/health')" || exit 1

CMD ["uvicorn", "iot_pipeline.api:app", "--host", "0.0.0.0", "--port", "8083"]
