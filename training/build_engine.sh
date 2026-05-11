#!/usr/bin/env bash
set -euo pipefail

MODELS_DIR=${1:-.}
HANDS=${2:-1}
INPUT_DIM=$((HANDS * 63))

echo "Building TensorRT engine for ${HANDS}-hand model (${INPUT_DIM} features)..."

trtexec --onnx="${MODELS_DIR}/model.onnx" \
        --saveEngine="${MODELS_DIR}/model.plan" \
        --fp16 \
        --minShapes="input:1x${INPUT_DIM}" \
        --optShapes="input:32x${INPUT_DIM}" \
        --maxShapes="input:64x${INPUT_DIM}"

echo "Engine saved to ${MODELS_DIR}/model.plan"
