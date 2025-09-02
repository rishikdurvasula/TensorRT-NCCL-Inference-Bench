#include "trt_runner.hpp"
#include "nccl_utils.hpp"
#include "common.hpp"
#include <iostream>
#include <cstdlib>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " --engine <path> [--batch 32] [--iters 500]\n";
        return 1;
    }
    std::string engine;
    int batch = 32;
    int iters = 500;
    for (int i=1;i<argc;i++) {
        std::string a = argv[i];
        if (a == "--engine" && i+1<argc) engine = argv[++i];
        else if (a == "--batch" && i+1<argc) batch = std::atoi(argv[++i]);
        else if (a == "--iters" && i+1<argc) iters = std::atoi(argv[++i]);
    }
    if (engine.empty()) { std::cerr << "--engine required\n"; return 1; }

    // NCCL world (optional; works as single process if WORLD_SIZE not set)
    NCCLContext nctx = ncclInitFromEnv();

    TRTRunner runner({engine, batch, iters, /*device*/ nctx.rank, true});
    runner.warmup(50);
    double ips = runner.runBenchmark();
    if (nctx.world_size > 1) {
        // Gather throughput to rank 0 via AllReduce sum
        float val = static_cast<float>(ips);
        float sum = 0.0f;
        ncclAllReduce(&val, &sum, 1, ncclFloat, ncclSum, nctx.comm, 0);
        cudaStreamSynchronize(0);
        if (nctx.rank == 0) {
            std::cout << "Aggregated throughput (images/sec): " << sum << std::endl;
        }
    } else {
        std::cout << "Throughput (images/sec): " << ips << std::endl;
    }
    return 0;
}
