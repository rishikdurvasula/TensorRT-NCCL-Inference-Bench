#!/usr/bin/env bash
set -euo pipefail
ENGINE=${1:-./data/resnet50_fp16.plan}
BATCH=${BATCH:-32}
ITERS=${ITERS:-500}
nsys profile -o nsys_report ./build/trt_infer --engine "$ENGINE" --batch "$BATCH" --iters "$ITERS"
ncu --set full --csv --target-processes all ./build/trt_infer --engine "$ENGINE" --batch "$BATCH" --iters "$ITERS"
