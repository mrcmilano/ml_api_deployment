# =========================
# Dockerfile - Produzione
# =========================

FROM python:3.11-slim

# Per disabilitare buffering dei logs
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiamo app e modelli nella directory di lavoro
COPY app/ ./app
COPY models/ ./models

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
