#!/usr/bin/env bash
set -euo pipefail

# Use a writable apt state/cache under /tmp to avoid read-only filesystem issues.
APT_DIR=/tmp/apt
mkdir -p "$APT_DIR/lists/partial" "$APT_DIR/cache/archives"

apt-get -o Dir::State="$APT_DIR" \
        -o Dir::State::Lists="$APT_DIR/lists" \
        -o Dir::State::status="$APT_DIR/status" \
        -o Dir::Cache="$APT_DIR/cache" \
        -o Dir::Cache::archives="$APT_DIR/cache/archives" update

apt-get -o Dir::State="$APT_DIR" \
        -o Dir::State::Lists="$APT_DIR/lists" \
        -o Dir::State::status="$APT_DIR/status" \
        -o Dir::Cache="$APT_DIR/cache" \
        -o Dir::Cache::archives="$APT_DIR/cache/archives" install -y \
        build-essential pkg-config libfftw3-dev liblapack-dev libblas-dev \
        libtag1-dev libyaml-dev libsamplerate0-dev libavcodec-dev \
        libavformat-dev libavutil-dev libswresample-dev

pip install --upgrade pip
pip install -r requirements.txt
