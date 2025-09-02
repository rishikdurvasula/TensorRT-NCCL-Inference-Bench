#!/usr/bin/env bash
set -euo pipefail
ENGINE=${1:-./data/resnet50_fp16.plan}
WORLD_SIZE=${WORLD_SIZE:-2}

# Simple local launcher: spawn $WORLD_SIZE processes bound to GPUs 0..WORLD_SIZE-1
for ((i=0;i<${WORLD_SIZE};i++)); do
  RANK=$i LOCAL_RANK=$i WORLD_SIZE=$WORLD_SIZE \
    ./build/trt_infer --engine "$ENGINE" --batch "${BATCH:-32}" --iters "${ITERS:-500}" &
done
wait
