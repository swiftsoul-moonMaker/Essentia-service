FROM python:3.9-slim

# Install system dependencies required by Essentia and FFmpeg
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      curl \
      build-essential pkg-config libfftw3-dev liblapack-dev libblas-dev \
      libtag1-dev libyaml-dev libsamplerate0-dev libavcodec-dev \
      libavformat-dev libavutil-dev libswresample-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY . .

# Ensure TensorFlow model files are present (download if missing in repo)
RUN mkdir -p /app/tensorflow-models && \
    if [ ! -f /app/tensorflow-models/msd-musicnn-1.pb ]; then \
      curl -L -o /app/tensorflow-models/msd-musicnn-1.pb \
        https://essentia.upf.edu/models/feature-extractors/musicnn/msd-musicnn-1.pb ; \
    fi && \
    if [ ! -f /app/tensorflow-models/msd-musicnn-1.json ]; then \
      curl -L -o /app/tensorflow-models/msd-musicnn-1.json \
        https://essentia.upf.edu/models/feature-extractors/musicnn/msd-musicnn-1.json ; \
    fi

# Environment defaults for TensorFlow tagger; override in Render if paths differ
ENV ESSENTIA_TF_MODEL=/app/tensorflow-models/msd-musicnn-1.pb
ENV ESSENTIA_TF_MODEL_LABELS=/app/tensorflow-models/msd-musicnn-1.json

# Render sets $PORT; fall back to 8000 for local use.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
