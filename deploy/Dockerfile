FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8501 \
    APP_HOST=0.0.0.0

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ship a reproducible Blue champion inside the public web image. The model is trained only on
# bounded synthetic observables; no Qwen endpoint or external data is required at build time.
RUN python scripts/train_detector.py \
    --seeds-per-cell 2 \
    --model-out data/loop/models/champion \
    --report /tmp/masterguard-detector-report.json

EXPOSE 8501

CMD ["gunicorn", "--bind", "0.0.0.0:8501", "--workers", "1", "--threads", "8", "--timeout", "300", "app.server:app"]
