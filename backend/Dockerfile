FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    HF_HOME=/tmp/hf \
    TRANSFORMERS_CACHE=/tmp/hf \
    PADDLE_HOME=/tmp/paddle \
    EASYOCR_MODULE_PATH=/tmp/easyocr

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    ffmpeg tesseract-ocr build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

COPY . .

RUN mkdir -p /app/uploads /app/outputs /tmp/hf /tmp/paddle /tmp/easyocr && \
    chmod -R 777 /app/uploads /app/outputs /tmp/hf /tmp/paddle /tmp/easyocr

EXPOSE 7860

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
