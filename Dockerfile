# Use NVIDIA's official TensorRT container with CUDA/cuDNN preinstalled
FROM nvcr.io/nvidia/tensorrt:24.04-py3

WORKDIR /workspace

# Install deps for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git wget python3-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Python deps (ONNX export + tools)
RUN python3 -m pip install --no-cache-dir torch torchvision onnx onnxruntime tqdm numpy pillow

COPY . /workspace
RUN chmod +x scripts/*.sh
