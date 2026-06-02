#!/usr/bin/env bash
set -euo pipefail

# Usage: build_engine.sh <models_dir> <hands>
# e.g.   build_engine.sh /languages/asl 1
#
# Deploys model.onnx into the Triton model repository.
# Requires /triton_repo to be mounted (e.g. -v $(pwd)/triton_repo:/triton_repo).

MODELS_DIR=${1:-.}
HANDS=${2:-1}

LANG=$(basename "$MODELS_DIR")                          # asl | isl
TRITON_DIR="${TRITON_REPO:-/triton_repo}/${LANG}_classifier/1"

echo "Deploying ${LANG} ONNX model to Triton repo..."
mkdir -p "$TRITON_DIR"
cp "${MODELS_DIR}/model.onnx" "${TRITON_DIR}/model.onnx"
echo "Deployed to ${TRITON_DIR}/model.onnx"
