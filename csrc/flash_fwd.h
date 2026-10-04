#pragma once
// Host-side interface of the FlashAttention-2 forward kernel. No PyTorch types here:
// bindings.cpp (Python) and dev/main.cu (standalone) both call flash_fwd().

#include <cuda_runtime.h>

#include <cstdint>

namespace fa {

// Which build of the kernel to run (see docs/optimizations.md, "Beyond FA2").
enum class Variant : std::uint8_t {
    kBaseline = 0,  // the first tuned FA2 kernel, kept as the reference point
    kOpt = 1,       // + FA3/FA4-style micro-optimizations; fp32 accumulation (default)
    kFp16Acc = 2,   // + P V accumulated in fp16 per tile (2x tensor rate on GeForce), opt-in
};

struct FlashFwdParams {
    const void* q = nullptr;  // [B, H, N, D] fp16, contiguous
    const void* k = nullptr;  // [B, H, N, D] fp16, contiguous
    const void* v = nullptr;  // [B, H, N, D] fp16, contiguous
    void* o = nullptr;        // [B, H, N, D] fp16, contiguous (output)
    int batch = 0;
    int heads = 0;
    int seqlen = 0;              // N (same for queries and keys)
    int head_dim = 0;            // D, 64 or 128
    float softmax_scale = 0.0F;  // usually 1/sqrt(D)
    bool causal = false;
    Variant variant = Variant::kOpt;
};

// Launches the kernel on `stream`. Throws std::runtime_error on bad arguments or CUDA errors.
void flash_fwd(const FlashFwdParams& params, cudaStream_t stream);

}  // namespace fa
