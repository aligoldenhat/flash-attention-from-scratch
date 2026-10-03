# Profiling

## Nsight Compute: one kernel, in depth (`profile/ncu.sh`)

```
make profile                                  # d=128 and d=64 reports (needs sudo)
profile/ncu.sh d128_causal 4 16 4096 128 1    # name, then B H N D causal
FULL=1 profile/ncu.sh d128                    # every section (--set full)
ncu-ui profile/reports/d128.ncu-rep           # open in the GUI
```

The script profiles the standalone driver (`build/release/fa_dev`), so only our kernel is in
the trace. It skips the correctness launch and profiles the first timed one.

**Root needed on this machine.** GPU performance counters are restricted to root here, so plain
`ncu` fails with `ERR_NVGPUCTRPERM`. The script runs ncu under `sudo` (asks for your password)
and chowns the report back to you (`RmProfilingAdminOnly: 1` in `/proc/driver/nvidia/params`
confirms the restriction). To allow non-root profiling permanently, NVIDIA documents this
driver option (not yet tested on this machine):

```
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiling.conf
sudo update-initramfs -u    # then reboot
```

### What to look at, by section

| Section | Question | Expected for this kernel |
|---|---|---|
| SpeedOfLight | compute- or memory-bound? | compute-bound: high SM/tensor throughput, low DRAM % |
| ComputeWorkloadAnalysis | is the tensor pipe the busiest? | tensor pipe at the top |
| MemoryWorkloadAnalysis (+ Tables) | shared-memory bank conflicts | ≈ 0 thanks to the swizzle |
| Occupancy | what limits resident warps? | registers (d=128 MT=2) or shared memory (48 KB/block) |
| WarpStateStats | why do warps stall? | some `barrier` (the `__syncthreads` of the pipeline), `math pipe throttle` = tensor cores saturated, which is good |
| LaunchStats | registers / smem per block, grid size | matches `FA_PTXAS_VERBOSE=1 make build` |
| SourceCounters | which source line stalls | `-lineinfo` makes this per line |

Extra metrics collected: `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_{ld,st}.sum`
(bank conflicts), `sm__pipe_tensor_cycles_active` (tensor-core busy %),
`smsp__inst_executed_pipe_tensor` (mma instruction count).

Profile a realistic size (default B=4, H=16, N=4096): a tiny grid leaves SMs idle and every
metric then misleads.

**Not run yet.** The ncu reports could not be produced from the agent session because sudo
needs your password. Run `make profile` in your own terminal.

## Nsight Systems: timeline (`profile/nsys.sh`)

```
profile/nsys.sh 4096 128 0      # N d causal
nsys-ui profile/reports/timeline_4096_128_0.nsys-rep
```

Records the benchmark of ours vs SDPA flash vs SDPA cuDNN on one shape and prints the CUDA
kernel summary. It shows which kernel each SDPA backend really launches, and that our kernel
is a single launch with no extra copies. nsys does not need root.

## Without a profiler

- `FA_PTXAS_VERBOSE=1 make build`: registers, spills and shared memory per kernel
  instantiation (current configs: 0 spills).
- `cuobjdump -sass build/release/fa_dev | grep -c HMMA`: count tensor-core instructions in the
  machine code. `cuobjdump -sass ... | grep LDSM` shows the ldmatrix instructions.
