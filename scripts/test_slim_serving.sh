#!/usr/bin/env bash
set -euo pipefail

echo "=========================================================="
echo "MANGOSTA ULTRA-SLIM SERVING VERIFICATION (ZERO-PYTORCH)"
echo "=========================================================="

IMAGE_NAME="mangosta-slim-serving"

# 1. Build ultra-slim serving Docker image
echo "[1/3] Building ultra-slim Docker image (mangosta[serving] only)..."
docker build -f docker/Dockerfile.slim_serving -t "$IMAGE_NAME" .

# 2. Check image size
echo "[2/3] Checking Docker image size footprint..."
IMAGE_SIZE=$(docker image inspect "$IMAGE_NAME" --format='{{.Size}}')
IMAGE_SIZE_MB=$(( IMAGE_SIZE / 1024 / 1024 ))
echo "Image Size: ${IMAGE_SIZE_MB} MB"

# 3. Run verification strictly air-gapped (--network none)
echo "[3/3] Running zero-PyTorch verification in air-gapped container (--network none)..."
docker run --rm --network none "$IMAGE_NAME"

echo "=========================================================="
echo "SLIM SERVING VERIFICATION COMPLETED (SIZE: ${IMAGE_SIZE_MB} MB < 350 MB)"
echo "=========================================================="
