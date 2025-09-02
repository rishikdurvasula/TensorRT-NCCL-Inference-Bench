#pragma once
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <memory>

struct TRTConfig {
    std::string engine_path;
    int batch = 32;
    int iters = 500;
    int device = 0;
    bool use_pinned = true;
};

class TRTRunner {
public:
    explicit TRTRunner(const TRTConfig& cfg);
    ~TRTRunner();
    void warmup(int iters);
    double runBenchmark(); // returns images/sec
    int inputSize() const;
    int outputSize() const;

private:
    TRTConfig cfg_;
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;
    cudaStream_t stream_ = 0;
    void* d_input_ = nullptr;
    void* d_output_ = nullptr;
    void* h_input_ = nullptr;
    void* h_output_ = nullptr;
    size_t input_bytes_ = 0;
    size_t output_bytes_ = 0;
    void allocate();
    void destroy();
    void prepareRandomInput();
};
