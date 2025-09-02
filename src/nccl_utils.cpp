#include "nccl_utils.hpp"
#include <cuda_runtime.h>
#include <cstdlib>

NCCLContext ncclInitFromEnv() {
    NCCLContext ctx;
    const char* ws = std::getenv("WORLD_SIZE");
    const char* rk = std::getenv("RANK");
    const char* ld = std::getenv("LOCAL_RANK");
    ctx.world_size = ws ? std::atoi(ws) : 1;
    ctx.rank = rk ? std::atoi(rk) : 0;
    int local_rank = ld ? std::atoi(ld) : 0;

    cudaSetDevice(local_rank);
    ncclUniqueId id;
    if (ctx.rank == 0) {
        ncclCheck(ncclGetUniqueId(&id), "get unique id");
        // Broadcast ID via tmp file or env in real use; for simplicity we assume single-node launch with mpirun that sets NCCL envs
        setenv("NCCL_UNIQUE_ID", (const char*)&id, 1); // placeholder; replace with proper launcher in practice
    } else {
        // In real multi-proc launch, NCCL_UNIQUE_ID is provided by rank 0 through mpirun or file share
    }
    const char* id_env = std::getenv("NCCL_UNIQUE_ID");
    if (!id_env) {
        // Fallback: single-process
        ctx.world_size = 1;
        ctx.rank = 0;
    }
    cudaStreamCreate(&ctx.stream);
    ncclCheck(ncclCommInitRank(&ctx.comm, ctx.world_size, id, ctx.rank), "comm init rank");
    return ctx;
}

void ncclSyncAll(NCCLContext& ctx) {
    // Dummy AllReduce to synchronize
    float one = 1.0f, acc = 0.0f;
    float* d_in=nullptr; float* d_out=nullptr;
    cudaMalloc(&d_in, sizeof(float));
    cudaMalloc(&d_out, sizeof(float));
    cudaMemcpyAsync(d_in, &one, sizeof(float), cudaMemcpyHostToDevice, ctx.stream);
    ncclCheck(ncclAllReduce(d_in, d_out, 1, ncclFloat, ncclSum, ctx.comm, ctx.stream), "allreduce");
    cudaStreamSynchronize(ctx.stream);
    cudaFree(d_in); cudaFree(d_out);
}
