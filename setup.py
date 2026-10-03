"""Builds the fa._C extension:  make build   (or: pip install -e . --no-build-isolation)

GPU architectures come from TORCH_CUDA_ARCH_LIST, e.g. "8.6" (RTX 3050), "8.9" (RTX 4090),
"12.0" (RTX 5090). If unset, PyTorch compiles for the GPU(s) it detects.
"""

import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

nvcc_flags = [
    "-O3",
    "-std=c++20",
    "--use_fast_math",  # exp2f -> ex2.approx (1 SFU instruction)
    "-lineinfo",  # SASS -> source mapping for ncu / compute-sanitizer, zero runtime cost
    "--expt-relaxed-constexpr",
]
if os.environ.get("FA_PTXAS_VERBOSE"):
    nvcc_flags.append("-Xptxas=-v")  # print registers / spills / smem per kernel

setup(
    name="fa",
    version="0.1.0",
    packages=["fa"],
    ext_modules=[
        CUDAExtension(
            name="fa._C",
            sources=["csrc/bindings.cpp", "csrc/kernels/flash_fwd.cu"],
            extra_compile_args={"cxx": ["-O3", "-std=c++20"], "nvcc": nvcc_flags},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
