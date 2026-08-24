FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8501 \
    APP_HOST=0.0.0.0

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["gunicorn", "--bind", "0.0.0.0:8501", "--workers", "1", "--threads", "8", "--timeout", "300", "app.server:app"]
