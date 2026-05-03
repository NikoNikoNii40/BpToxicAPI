# --- stage: downloader ---
FROM gcr.io/google.com/cloudsdktool/cloud-sdk:slim AS downloader
WORKDIR /dl
ARG MODEL_GCS=gs://bp-toxicapi-models/models/xlmr-toxic-v2_1
RUN gsutil -m cp -r ${MODEL_GCS} /dl/xlmr-toxic-v2_1

# --- stage: runtime ---
FROM python:3.12-slim
WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY api /app/api
COPY --from=downloader /dl/xlmr-toxic-v2_1 /app/models/xlmr-toxic-v2_1

ENV HF_MODEL_PATH=/app/models/xlmr-toxic-v2_1 \
    MODEL_ID=xlmr-toxic-v2_1 \
    THRESHOLDS_PATH=/app/api/thresholds_product_v2_1.json \
    THRESHOLD_SET=product_v2_1

CMD ["sh", "-c", "uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8080}"]