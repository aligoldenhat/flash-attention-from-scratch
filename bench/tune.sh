#!/usr/bin/env bash
# Sweep tile configurations for one (head dim, causal) slot and print TFLOPS for each.
#
#   bench/tune.sh <d> <causal> [warps,MT,BC,QinRegs ...]
#   bench/tune.sh 128 0                        # default candidate list
#   bench/tune.sh 64 1 4,1,64,true 4,2,64,true
#
# For each candidate it rewrites the matching `using ConfigD<d>[Causal] = Config<...>;` line
# in csrc/kernels/flash_fwd.cu, rebuilds the standalone driver (build/release/fa_dev) and runs
# it on FA2-paper shapes (B*N = 16k tokens, H*d = 2048), which also checks correctness.
# Each cell is the best of 3 medians.
# The source file is restored afterwards; copy the winning line in by hand.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ $# -lt 2 ]]; then
    sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//' >&2
    exit 2
fi
d=$1
causal=$2
shift 2
candidates=("$@")
if [[ ${#candidates[@]} -eq 0 ]]; then
    candidates=(4,1,64,true 4,1,64,false 4,1,32,true 4,2,32,true 4,2,32,false 4,2,64,true
                4,2,64,false 8,1,64,true 8,1,32,true)
fi
name="ConfigD${d}"
if [[ $causal == 1 ]]; then name+="Causal"; fi

src=csrc/kernels/flash_fwd.cu
backup=$(mktemp)
cp "$src" "$backup"
trap 'cp "$backup" "$src"; rm -f "$backup"' EXIT

ns=(512 1024 4096 16384)
cmake --preset release >/dev/null
printf '%-18s' "$name"
printf ' N=%-6s' "${ns[@]}"
printf '  (TFLOPS)\n'
for cand in "${candidates[@]}"; do
    sed -i -E "s/^using ${name} = Config<.*>;/using ${name} = Config<${d}, ${cand//,/, }>;/" "$src"
    printf '%-18s' "$cand"
    if ! cmake --build --preset release >/dev/null 2>&1; then
        echo " build failed"
        continue
    fi
    for n in "${ns[@]}"; do
        # Best of 3 runs (each a median of 50 launches): laptop GPU clocks are noisy.
        best=""
        for _ in 1 2 3; do
            line=$(./build/release/fa_dev $((16384 / n)) $((2048 / d)) "$n" "$d" "$causal" 50 2>&1 || true)
            if [[ $line == *FAILED* || $line != *TFLOPS* ]]; then
                best=FAIL
                break
            fi
            tf=$(echo "$line" | grep -oE '[0-9.]+ TFLOPS' | cut -d' ' -f1)
            best=$(awk -v a="$best" -v b="$tf" 'BEGIN { print (a == "" || b > a) ? b : a }')
        done
        printf ' %-8s' "$best"
    done
    echo
done
