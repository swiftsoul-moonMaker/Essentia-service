FROM python:3.9-slim

# Install system dependencies required by Essentia and FFmpeg
RUN apt-get update && DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
        curl build-essential pkg-config libfftw3-dev \
        liblapack-dev libblas-dev \
        libtag1-dev libyaml-dev libsamplerate0-dev \
        libavcodec-dev libavformat-dev libavutil-dev libswresample-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install numpy==1.26.4 fastapi uvicorn[standard] pydantic python-multipart

RUN curl -L \
      "https://files.pythonhosted.org/packages/9e/30/8e3306ef13584cf3e4fcd7c2430c5c592c23c23bc65d58c0fd6e22ee6c73/essentia-2.1b6.dev1389-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl" \
      -o /tmp/essentia.whl \
   && curl -L \
      "https://files.pythonhosted.org/packages/21/c1/302af923c626b90e4f5ef0cfae4a8f97cb65f00f8d8cb646cf797e72f696/essentia_tensorflow-2.1b6.dev1389-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl" \
      -o /tmp/essentia_tf.whl \
   && pip install /tmp/essentia.whl /tmp/essentia_tf.whl \
   && rm -f /tmp/essentia.whl /tmp/essentia_tf.whl

COPY . .

# Environment defaults for TensorFlow tagger; override in Render if paths differ
ENV ESSENTIA_TF_MODEL=/app/tensorflow-models/msd-musicnn-1.pb
ENV ESSENTIA_TF_MODEL_LABELS=/app/tensorflow-models/msd-musicnn-1.json

# Render sets $PORT; fall back to 8000 for local use.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
