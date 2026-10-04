#!/usr/bin/env bash
# Ablation: measure each optional kernel feature on its own (and combinations).
#
#   bench/ablate.sh <opt|fp16acc> "PEEL,LAZY,EXP16,EMU,ACC16" ...
#   bench/ablate.sh opt "false,false,false,false,false" "true,false,false,false,false"
# (STAGES is part of the tile config: tune it with bench/tune.sh)
#
# For each flag set it rewrites the `using OptOpt = Opt<...>;` (or FastOpt for fp16acc) line
# in csrc/kernels/flash_fwd.cu, rebuilds the standalone driver and runs it on FA2-paper shapes
# (B*N = 16k tokens, H*d = 2048) for every (d, causal) slot, using that variant's tiles.
# Each cell is the best of 3 medians; every run also checks correctness against the CPU.
# The source file is restored afterwards.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ $# -lt 2 ]]; then
    sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//' >&2
    exit 2
fi
variant=$1
shift
case $variant in
    opt) line=OptOpt ;;
    fp16acc) line=FastOpt ;;
    *) echo "variant must be opt or fp16acc" >&2; exit 2 ;;
esac

src=csrc/kernels/flash_fwd.cu
backup=$(mktemp)
cp "$src" "$backup"
trap 'cp "$backup" "$src"; rm -f "$backup"' EXIT

ns=(1024 4096)
cmake --preset release >/dev/null
printf '%-36s' "$line (PEEL,LAZY,EXP16,EMU,ACC16)"
for d in 64 128; do for c in 0 1; do for n in "${ns[@]}"; do
    printf ' %-9s' "d${d}c${c}N$((n / 1024))k"
done; done; done
echo
for flags in "$@"; do
    sed -i -E "s/^using ${line} *= Opt<.*>;/using ${line} = Opt<${flags//,/, }>;/" "$src"
    printf '%-36s' "$flags"
    if ! cmake --build --preset release >/dev/null 2>&1; then
        echo " build failed"
        continue
    fi
    for d in 64 128; do for causal in 0 1; do for n in "${ns[@]}"; do
        best=""
        for _ in 1 2 3; do
            out=$(./build/release/fa_dev $((16384 / n)) $((2048 / d)) "$n" "$d" "$causal" 30 "$variant" 2>&1 || true)
            if [[ $out == *FAILED* || $out != *TFLOPS* ]]; then
                best=FAIL
                break
            fi
            tf=$(echo "$out" | grep -oE '[0-9.]+ TFLOPS' | cut -d' ' -f1)
            best=$(awk -v a="$best" -v b="$tf" 'BEGIN { print (a == "" || b > a) ? b : a }')
        done
        printf ' %-9s' "$best"
    done; done; done
    echo
done
