#pragma once
#include <nccl.h>
#include <string>
#include <vector>
#include <iostream>

inline void ncclCheck(ncclResult_t r, const char* msg) {
    if (r != ncclSuccess) {
        std::cerr << "NCCL Error (" << msg << "): " << ncclGetErrorString(r) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

struct NCCLContext {
    int world_size = 1;
    int rank = 0;
    ncclComm_t comm;
    cudaStream_t stream = 0;
};

NCCLContext ncclInitFromEnv();
void ncclSyncAll(NCCLContext& ctx);
