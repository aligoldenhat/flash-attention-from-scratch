// PyTorch bindings: validates tensors, then calls the torch-free launcher in flash_fwd.cu.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cmath>
#include <optional>
#include <string>

#include "flash_fwd.h"

namespace {

fa::Variant parse_variant(const std::string& name) {
    if (name == "opt") {
        return fa::Variant::kOpt;
    }
    if (name == "fp16acc") {
        return fa::Variant::kFp16Acc;
    }
    if (name == "baseline") {
        return fa::Variant::kBaseline;
    }
    TORCH_CHECK(false, "variant must be 'opt', 'fp16acc' or 'baseline', got '", name, "'");
}

torch::Tensor forward(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
                      bool causal, std::optional<double> softmax_scale,
                      const std::string& variant) {
    for (const auto* t : {&q, &k, &v}) {
        TORCH_CHECK(t->is_cuda(), "q, k, v must be CUDA tensors");
        TORCH_CHECK(t->scalar_type() == torch::kHalf, "q, k, v must be float16");
        TORCH_CHECK(t->is_contiguous(), "q, k, v must be contiguous");
        TORCH_CHECK(t->dim() == 4, "q, k, v must have shape [B, H, N, D]");
    }
    TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(),
                "q, k, v must have the same shape");
    TORCH_CHECK(q.device() == k.device() && q.device() == v.device(),
                "q, k, v must be on the same device");

    const c10::cuda::CUDAGuard guard(q.device());
    auto o = torch::empty_like(q);

    const auto head_dim = q.size(3);
    fa::FlashFwdParams p{};
    p.q = q.data_ptr();
    p.k = k.data_ptr();
    p.v = v.data_ptr();
    p.o = o.data_ptr();
    p.batch = static_cast<int>(q.size(0));
    p.heads = static_cast<int>(q.size(1));
    p.seqlen = static_cast<int>(q.size(2));
    p.head_dim = static_cast<int>(head_dim);
    p.softmax_scale =
        static_cast<float>(softmax_scale.value_or(1.0 / std::sqrt(static_cast<double>(head_dim))));
    p.causal = causal;
    p.variant = parse_variant(variant);

    if (p.seqlen > 0) {
        fa::flash_fwd(p, at::cuda::getCurrentCUDAStream());
    }
    return o;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashAttention-2 forward pass written from scratch (mma.sync, cp.async, swizzle)";
    m.def("forward", &forward, "FlashAttention forward: O = softmax(Q K^T * scale) V", py::arg("q"),
          py::arg("k"), py::arg("v"), py::arg("causal") = false,
          py::arg("softmax_scale") = py::none(), py::arg("variant") = "opt");
}
