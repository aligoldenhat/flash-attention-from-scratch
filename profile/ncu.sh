#!/usr/bin/env bash
# Nsight Compute report of the attention kernel, with the sections that matter for it.
#
#   profile/ncu.sh [name] [B H N D causal]      default: d128 4 16 4096 128 0
#   profile/ncu.sh d64_causal 4 32 4096 64 1
#   FULL=1 profile/ncu.sh ...                   # --set full instead (slower, everything)
#
# Writes profile/reports/<name>.ncu-rep (open with ncu-ui) and a text summary next to it.
# Runs the standalone driver build/release/fa_dev, so only our kernel is in the trace.
#
# Sections:
#   SpeedOfLight            % of peak compute / memory: the first thing to look at
#   ComputeWorkloadAnalysis tensor-pipe utilisation (is mma.sync the busiest pipe?)
#   MemoryWorkloadAnalysis  DRAM / L2 / shared traffic, incl. shared bank conflicts (tables)
#   Occupancy               theoretical vs achieved; limiter: registers or shared memory
#   WarpStateStats          why warps stall (barrier, long scoreboard, mio throttle, ...)
#   LaunchStats             grid, registers per thread, shared memory per block
#   SourceCounters          per-line stalls and bank conflicts (needs -lineinfo: on)
#
# ncu needs GPU performance counters; on this machine they are root-only
# (ERR_NVGPUCTRPERM), so ncu runs under sudo and the report is chown'ed back to you.
set -euo pipefail
cd "$(dirname "$0")/.."

NAME=${1:-d128}
shift || true
ARGS=("$@")
if [[ ${#ARGS[@]} -eq 0 ]]; then
    ARGS=(4 16 4096 128 0)
fi
ARGS+=(1)  # iterations: the driver runs the kernel once for checking + once timed

BIN=build/release/fa_dev
if [[ ! -x $BIN ]]; then
    cmake --preset release >/dev/null && cmake --build --preset release
fi
mkdir -p profile/reports
REPORT=profile/reports/$NAME

if [[ -n ${FULL:-} ]]; then
    SET=(--set full)
else
    SET=()
    for s in SpeedOfLight ComputeWorkloadAnalysis MemoryWorkloadAnalysis \
        MemoryWorkloadAnalysis_Tables Occupancy WarpStateStats LaunchStats SourceCounters; do
        SET+=(--section "$s")
    done
fi

NCU=$(command -v ncu)
SUDO=()
if [[ $EUID -ne 0 ]]; then SUDO=(sudo); fi

# --launch-skip 1: skip the correctness launch, profile the first timed one.
"${SUDO[@]}" "$NCU" "${SET[@]}" --import-source on \
    --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,smsp__inst_executed_pipe_tensor.sum,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active \
    --kernel-name "regex:flash_fwd_kernel" --launch-skip 1 --launch-count 1 \
    --force-overwrite -o "$REPORT" "$BIN" "${ARGS[@]}"
"${SUDO[@]}" chown "$USER" "$REPORT.ncu-rep"

ncu --import "$REPORT.ncu-rep" --print-summary none > "$REPORT.txt"
echo "Report:  $REPORT.ncu-rep   (ncu-ui $REPORT.ncu-rep)"
echo "Summary: $REPORT.txt"
