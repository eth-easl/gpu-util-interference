# Pitfall: Colocating memory and compute bound kernels
This folder contains the experimental code supporting Pitfall 1 in Section 3 of our [report](https://arxiv.org/pdf/2501.16909). The benchmark shows that colocating compute and memory-bound kernels may not always improve performance, as it does not account for per-SM instruction issue rates (IPC).

The main script can be run as
```bash
./comp_mem_ipc <mode> <num_threads_per_tb> <num_itrs_comp> <num_itrs_copy>
```
#### Parameters
- `mode`: Execution mode for benchmark (0: profiling, 1: run alone, 2: run sequential, 3: run collocated)
- `num_threads_per_tb`: number of threads per block for the compute and copy kernel
- `num_itrs_comp`: number of iterations for the compute kernel
- `num_itrs_copy`: number of iterations for the copy kernel

### Running the paper experiment
```bash
sh run_ipc_experiment.sh
```
The experiment compares:
- Individual latencies of the compute and copy kernels running in isolation
- Makespan when both kernels run colocated using CUDA streams

**Note**: Current parameters in the `run_ipc_experiment.sh` are tailored to the H100 GPU. For other GPUs adjust the valuesas follows:
- `num_threads_per_tb` should be set to `max_threads_per_multiprocessor / 2`. This allows one thread block of the compute and copy kernel to run on the same SM without exceeding the maximum number of threads per SM.
- `num_iters_comp` and `num_iters_mem` should be adjusted such that both kernels have similar runtimes when run in isolation (mode = 1).

### Relevant NCU metrics
Depsite both kernels having complementary resource requirements (compute vs. memory-bound), both kernels may interfere due to saturating the warp scheduler. The relevant NCU metrics for this experiment are:
1. Average issued instructions per cycle per SM while active [instr/cycle]
    - `sm__inst_issued.avg.per_cycle_active`
2. Average issue slots busy [%]
    - `sm__inst_issued.avg.pct_of_peak_sustained_active`
