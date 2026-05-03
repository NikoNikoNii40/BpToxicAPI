FROM python:3.12-slim
WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY api /app/api
COPY models /app/models

ENV HF_MODEL_PATH=/app/models/xlmr-toxic-v2_1 \
    MODEL_ID=xlmr-toxic-v2_1 \
    THRESHOLDS_PATH=/app/api/thresholds_product_v2_1.json \
    THRESHOLD_SET=product_v2_1

CMD ["sh", "-c", "uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8080}"]