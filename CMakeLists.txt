cmake_minimum_required(VERSION 3.20)
project(trt_nccl_infer_bench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)

# Adjust these paths for native builds if needed
# set(TENSORRT_ROOT /usr/lib/x86_64-linux-gnu)
# set(NCCL_ROOT /usr/lib/x86_64-linux-gnu)

include_directories(
  ${CUDA_INCLUDE_DIRS}
  /usr/include
  /usr/local/cuda/include
  /usr/include/x86_64-linux-gnu
)

link_directories(
  /usr/lib/x86_64-linux-gnu
  /usr/local/cuda/lib64
)

add_executable(trt_infer
  src/main.cpp
  src/trt_runner.cpp
  src/nccl_utils.cpp
)

target_include_directories(trt_infer PRIVATE include)
target_compile_definitions(trt_infer PRIVATE -D_GLIBCXX_USE_CXX11_ABI=1)

# Link TensorRT, CUDA, cuBLAS, cuDNN, and NCCL
target_link_libraries(trt_infer
  nvinfer
  nvinfer_plugin
  cudart
  cublas
  cublasLt
  cudnn
  nccl
  pthread
)
