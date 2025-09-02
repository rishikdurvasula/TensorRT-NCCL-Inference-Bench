#include "trt_runner.hpp"
#include "common.hpp"
#include <fstream>
#include <random>
#include <cstring>

using namespace nvinfer1;

static std::vector<char> loadFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Failed to open engine: " << path << std::endl; std::exit(1); }
    return std::vector<char>(std::istreambuf_iterator<char>(f), {});
}

TRTRunner::TRTRunner(const TRTConfig& cfg): cfg_(cfg) {
    cudaCheck(cudaSetDevice(cfg_.device), "set device");
    runtime_ = createInferRuntime(gLogger);
    auto blob = loadFile(cfg_.engine_path);
    engine_ = runtime_->deserializeCudaEngine(blob.data(), blob.size());
    if (!engine_) { std::cerr << "Failed to deserialize engine\n"; std::exit(1); }
    context_ = engine_->createExecutionContext();
    cudaCheck(cudaStreamCreate(&stream_), "create stream");
    allocate();
}

TRTRunner::~TRTRunner() { destroy(); }

void TRTRunner::allocate() {
    int input_idx = engine_->getBindingIndex(engine_->getBindingName(0));
    int output_idx = engine_->getBindingIndex(engine_->getBindingName(1));
    auto in_dims = engine_->getBindingDimensions(input_idx);
    auto out_dims = engine_->getBindingDimensions(output_idx);
    // Assume NCHW input, float32
    int64_t in_elems = 1;
    for (int i=0;i<in_dims.nbDims;i++) in_elems *= in_dims.d[i];
    int64_t out_elems = 1;
    for (int i=0;i<out_dims.nbDims;i++) out_elems *= out_dims.d[i];
    input_bytes_ = sizeof(float) * cfg_.batch * in_elems;
    output_bytes_ = sizeof(float) * cfg_.batch * out_elems;

    if (cfg_.use_pinned) {
        cudaCheck(cudaMallocHost(&h_input_, input_bytes_), "pinned malloc host input");
        cudaCheck(cudaMallocHost(&h_output_, output_bytes_), "pinned malloc host output");
    } else {
        h_input_ = malloc(input_bytes_);
        h_output_ = malloc(output_bytes_);
    }
    cudaCheck(cudaMalloc(&d_input_, input_bytes_), "malloc device input");
    cudaCheck(cudaMalloc(&d_output_, output_bytes_), "malloc device output");
}

void TRTRunner::destroy() {
    if (d_input_) cudaFree(d_input_);
    if (d_output_) cudaFree(d_output_);
    if (cfg_.use_pinned) {
        if (h_input_) cudaFreeHost(h_input_);
        if (h_output_) cudaFreeHost(h_output_);
    } else {
        if (h_input_) free(h_input_);
        if (h_output_) free(h_output_);
    }
    if (context_) context_->destroy();
    if (engine_) engine_->destroy();
    if (runtime_) runtime_->destroy();
    if (stream_) cudaStreamDestroy(stream_);
}

int TRTRunner::inputSize() const { return (int)input_bytes_; }
int TRTRunner::outputSize() const { return (int)output_bytes_; }

void TRTRunner::prepareRandomInput() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    float* ptr = reinterpret_cast<float*>(h_input_);
    size_t elems = input_bytes_ / sizeof(float);
    for (size_t i=0;i<elems;i++) ptr[i] = dist(rng);
}

void TRTRunner::warmup(int iters) {
    void* bindings[2] = { d_input_, d_output_ };
    for (int i=0;i<iters;i++) {
        prepareRandomInput();
        cudaCheck(cudaMemcpyAsync(d_input_, h_input_, input_bytes_, cudaMemcpyHostToDevice, stream_), "H2D");
        bool ok = context_->enqueueV2(bindings, stream_, nullptr);
        if (!ok) { std::cerr << "enqueueV2 failed\n"; std::exit(1); }
        cudaCheck(cudaMemcpyAsync(h_output_, d_output_, output_bytes_, cudaMemcpyDeviceToHost, stream_), "D2H");
        cudaCheck(cudaStreamSynchronize(stream_), "sync");
    }
}

double TRTRunner::runBenchmark() {
    void* bindings[2] = { d_input_, d_output_ };
    Timer t; t.start();
    double total_ms = 0.0;
    int iters = cfg_.iters;
    for (int i=0;i<iters;i++) {
        prepareRandomInput();
        auto t0 = std::chrono::high_resolution_clock::now();
        cudaCheck(cudaMemcpyAsync(d_input_, h_input_, input_bytes_, cudaMemcpyHostToDevice, stream_), "H2D");
        bool ok = context_->enqueueV2(bindings, stream_, nullptr);
        if (!ok) { std::cerr << "enqueueV2 failed\n"; std::exit(1); }
        cudaCheck(cudaMemcpyAsync(h_output_, d_output_, output_bytes_, cudaMemcpyDeviceToHost, stream_), "D2H");
        cudaCheck(cudaStreamSynchronize(stream_), "sync");
        auto t1 = std::chrono::high_resolution_clock::now();
        total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    double avg_ms = total_ms / iters;
    double imgs_per_sec = (1000.0 / avg_ms) * cfg_.batch;
    return imgs_per_sec;
}
