#!/usr/bin/env bash
# Nsight Systems timeline of the benchmark: ours vs torch SDPA back-to-back on one shape.
#
#   profile/nsys.sh [N] [d] [causal]          default: 4096 128 0
#
# Writes profile/reports/timeline_<N>_<d>_<causal>.nsys-rep (open with nsys-ui) and prints
# the CUDA kernel summary (time per kernel name), which shows which kernel each SDPA
# backend actually launches (e.g. pytorch_flash::flash_fwd_kernel for the FA2 backend).
# nsys does not need root for CUDA tracing.
set -euo pipefail
cd "$(dirname "$0")/.."

N=${1:-4096}
D=${2:-128}
CAUSAL=${3:-0}
mkdir -p profile/reports
OUT=profile/reports/timeline_${N}_${D}_${CAUSAL}
PY=${PYTHON:-.venv/bin/python}

nsys profile --trace=cuda,nvtx --force-overwrite=true -o "$OUT" \
    "$PY" bench/benchmark.py --n "$N" --d "$D" --causal "$CAUSAL" \
    --impl "fa (ours)" "sdpa-flash (FA2)" "sdpa-cudnn" --out "$OUT.csv"
nsys stats --report cuda_gpu_kern_sum --format table "$OUT.nsys-rep" | head -25
