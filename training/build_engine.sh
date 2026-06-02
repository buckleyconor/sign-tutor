#!/usr/bin/env bash
set -euo pipefail

# Build a TensorRT FP16 engine from an ONNX model and deploy it to the
# Triton model repository.
#
# Usage: build_engine.sh <models_dir>
#        e.g. build_engine.sh languages/asl
#
# Requirements:
#   - /triton_repo must be mounted (e.g. -v $(pwd)/triton_repo:/triton_repo)
#   - trtexec must be available in PATH (comes with TensorRT in the
#     nvcr.io/nvidia/tritonserver container)
#
# The ONNX model is expected at $MODELS_DIR/model.onnx and the resulting
# TensorRT engine (.plan) is deployed to the Triton model repo.

MODELS_DIR=${1:?Usage: build_engine.sh <models_dir>}
TRITON_REPO=${TRITON_REPO:-/triton_repo}

LANG=$(basename "$MODELS_DIR")
ONNX_PATH="${MODELS_DIR}/model.onnx"
TRITON_DIR="${TRITON_REPO}/${LANG}_classifier/1"

if [ ! -f "$ONNX_PATH" ]; then
    echo "ERROR: ONNX model not found at ${ONNX_PATH}"
    echo "       Train/export the model first: python training/export_onnx.py"
    exit 1
fi

echo "=== Building TensorRT engine for ${LANG} ==="
echo "  ONNX: ${ONNX_PATH}"
echo "  Deploy: ${TRITON_DIR}/model.plan"

mkdir -p "${MODELS_DIR}"
mkdir -p "${TRITON_DIR}"

# Build the TensorRT FP16 engine with dynamic batch shapes.
# The engine encodes the architecture and precision; Triton handles
# dynamic batching via the config.pbtxt dynamic_batching directive.
trtexec \
    --onnx="${ONNX_PATH}" \
    --saveEngine="${MODELS_DIR}/model.plan" \
    --fp16 \
    --minShapes=input:1x63 \
    --optShapes=input:32x63 \
    --maxShapes=input:64x63 \
    --useCudaGraph \
    --timingCacheFile="${MODELS_DIR}/timing.cache" \
    --verbose 2>&1 | tee "${MODELS_DIR}/trt_build.log"

if [ ! -f "${MODELS_DIR}/model.plan" ]; then
    echo "ERROR: TensorRT engine build failed — no model.plan produced"
    exit 1
fi

ENGINE_SIZE=$(stat -c%s "${MODELS_DIR}/model.plan" 2>/dev/null || stat -f%z "${MODELS_DIR}/model.plan" 2>/dev/null)
echo "  Engine built: $(numfmt --to=iec ${ENGINE_SIZE} 2>/dev/null || echo "${ENGINE_SIZE}B")"

# Deploy to Triton model repository
cp "${MODELS_DIR}/model.plan" "${TRITON_DIR}/model.plan"
echo "  Deployed to ${TRITON_DIR}/model.plan"

# Clean up timing cache (per-engine, per-machine)
rm -f "${MODELS_DIR}/timing.cache"

echo "=== Done. ==="
echo "  Restart Triton or wait for poll reload to pick up the new engine."
echo "  Verify with: curl http://localhost:8000/v2/models/${LANG}_classifier/ready"
