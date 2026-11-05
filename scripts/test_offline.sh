#!/usr/bin/env bash
set -euo pipefail

echo "=========================================================="
echo "MANGOSTA OFFLINE AUDIT HARNESS (UBUNTU 24.04 AIR-GAPPED)"
echo "=========================================================="

IMAGE_NAME="mangosta-offline-test"

# 1. Build Docker image with latest working tree
echo "[1/2] Building test Docker image with current code..."
docker build -f docker/Dockerfile.offline_test -t "$IMAGE_NAME" .

# 2. Run container with zero internet access
echo "[2/2] Running verification suite strictly air-gapped (--network none)..."
docker run --rm --network none "$IMAGE_NAME"

echo "=========================================================="
echo "OFFLINE AUDIT HARNESS COMPLETED SUCCESSFULLY (100% PASS)"
echo "=========================================================="
