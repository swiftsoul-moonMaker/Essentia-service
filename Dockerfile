FROM python:3.10-slim

# Install system dependencies required by Essentia and FFmpeg
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential pkg-config libfftw3-dev liblapack-dev libblas-dev \
      libtag1-dev libyaml-dev libsamplerate0-dev libavcodec-dev \
      libavformat-dev libavutil-dev libswresample-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Environment defaults for TensorFlow tagger; override in Render if paths differ
ENV ESSENTIA_TF_MODEL=/app/tensorflow-models/msd-musicnn-1.pb
ENV ESSENTIA_TF_MODEL_LABELS=/app/tensorflow-models/msd-musicnn-1.json

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
