#!/usr/bin/env bash
set -euo pipefail
ENGINE=${1:-./data/resnet50_fp16.plan}
BATCH=${BATCH:-32}
ITERS=${ITERS:-500}
./build/trt_infer --engine "$ENGINE" --batch "$BATCH" --iters "$ITERS"
