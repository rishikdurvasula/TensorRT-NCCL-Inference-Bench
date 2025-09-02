# TRT-NCCL-Infer-Bench
High-performance **TensorRT** inference benchmark with **multi-GPU scaling via NCCL**, CUDA streams, and optional FP16/INT8.
Includes **Nsight Systems/Compute** profiling workflows and **cuBLAS/cuDNN** referential notes.




## Project Layout
```
trt-nccl-infer-bench/
  include/
    common.hpp
  src/
    main.cpp
    trt_runner.cpp
    trt_runner.hpp
    nccl_utils.cpp
    nccl_utils.hpp
    calibrator.hpp
  scripts/
    build.sh
    bench_single_gpu.sh
    bench_multi_gpu.sh
    export_onnx_resnet50.py
    build_trt_engine.py
    profile_nsight.sh
  configs/
    engine_config.json
  Dockerfile
  CMakeLists.txt
  README.md
```

## Requirements
- NVIDIA GPU + recent driver
- Docker with **nvidia-container-toolkit**, or native dev env with:
  - CUDA Toolkit (12.x recommended)
  - TensorRT SDK (>= 8.6)
  - cuDNN (bundled in base container)
  - NCCL (>= 2.18)
  - CMake >= 3.20, gcc/g++ >= 9

## Quick Start (Docker – recommended)
```bash
# 1) Pull NVIDIA TensorRT container
docker pull nvcr.io/nvidia/tensorrt:24.04-py3

# 2) Build project image
docker build -t trt-nccl-infer-bench .

# 3) Run container with GPU access
docker run --gpus all -it --rm -v $PWD:/workspace trt-nccl-infer-bench /bin/bash

# Inside the container:
# 4) Build
./scripts/build.sh

# 5) Export ONNX (ResNet50) and build TRT engines (FP32/FP16)
python3 ./scripts/export_onnx_resnet50.py --out ./data/resnet50.onnx
python3 ./scripts/build_trt_engine.py --onnx ./data/resnet50.onnx --precision fp16 --out ./data/resnet50_fp16.plan
python3 ./scripts/build_trt_engine.py --onnx ./data/resnet50.onnx --precision fp32 --out ./data/resnet50_fp32.plan

# 6) Single-GPU benchmark
./scripts/bench_single_gpu.sh ./data/resnet50_fp16.plan

# 7) Multi-GPU benchmark (2 GPUs example)
WORLD_SIZE=2 ./scripts/bench_multi_gpu.sh ./data/resnet50_fp16.plan
```

## Nsight Profiling
```bash
# Systems (timeline, kernel launches, CPU↔GPU overlap)
nsys profile -o nsys_report ./build/trt_infer --engine ./data/resnet50_fp16.plan --batch 32 --iters 500

# Compute (kernel metrics, tensor core utilization)
ncu --set full --csv --target-processes all ./build/trt_infer --engine ./data/resnet50_fp16.plan --batch 32 --iters 500
```
Inspect for:
- **SM occupancy**, **warp stall reasons** (selected/long scoreboard), **L2/TEX/L1 hit rates**
- **Tensor Core utilization** (HMMA kernels), **DRAM throughput**, **shared-memory bank conflicts**
- Host↔Device overlap and **memcpy** timings; ensure **pinned memory** and **async** copies.

## Reproducible Config
Edit `configs/engine_config.json`:
- Min/opt/max shapes, workspace size, precision, DLA, int8 calibration cache path.

## Building Without Docker
Set `TENSORRT_ROOT`, `CUDA_HOME`, and `NCCL_ROOT` and adjust `CMakeLists.txt` paths if needed.



