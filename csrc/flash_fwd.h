#pragma once
// Host-side interface of the FlashAttention-2 forward kernel. No PyTorch types here:
// bindings.cpp (Python) and dev/main.cu (standalone) both call flash_fwd().

#include <cuda_runtime.h>

namespace fa {

struct FlashFwdParams {
    const void* q;  // [B, H, N, D] fp16, contiguous
    const void* k;  // [B, H, N, D] fp16, contiguous
    const void* v;  // [B, H, N, D] fp16, contiguous
    void* o;        // [B, H, N, D] fp16, contiguous (output)
    int batch;
    int heads;
    int seqlen;           // N (same for queries and keys)
    int head_dim;         // D, 64 or 128
    float softmax_scale;  // usually 1/sqrt(D)
    bool causal;
};

// Launches the kernel on `stream`. Throws std::runtime_error on bad arguments or CUDA errors.
void flash_fwd(const FlashFwdParams& params, cudaStream_t stream);

}  // namespace fa
