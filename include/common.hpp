#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <chrono>
#include <iostream>
#include <memory>
#include <cassert>

inline void cudaCheck(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error (" << msg << "): " << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

struct Timer {
    std::chrono::high_resolution_clock::time_point t0;
    void start() { t0 = std::chrono::high_resolution_clock::now(); }
    double ms() const {
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
};
