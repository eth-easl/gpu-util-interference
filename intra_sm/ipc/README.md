# Issued Instructions Per Cycle (IPC) interference
This folder contains the experimental code support for section 4.2.2 of our [report](https://arxiv.org/pdf/2501.16909) demonstrating how kernel colocation can lead to warp scheduler interference.

The main script can be run as follows:
```bash
./ipc <mode> <kernel_id> <num_threads_per_tb_copy> <num_threads_per_tb_comp> <num_itrs_copy> <num_itrs_comp> <num_bytes>
```
#### Parameters
- `mode`: Execution mode for benchmark (0: profile, 0: run alone, 2: run sequential, 3: run collocated)
- `kernel_id`: ID of kernel to execute (0: copy, 1: ilp1, 2: ilp2, 3: ilp3, 4: ilp4)
- `num_threads_per_tb_copy`: Threads per block for copy kernel
- `num_threads_per_tb_comp`: Threads per block for compute kernel
- `num_itrs_copy`: Iterations for copy kernel
- `num_itrs_comp`: Iterations for compute kernel
- `num_bytes`: Number of bytes to be copied by the copy kernel

### Running the paper experiment
```bash
sh run_ipc_experiment.sh
```
The script compares latency when
- running copy and compute (`mul_fp32_ilp*`) kernels alone
- running copy and compute (`mul_fp32_ilp*`) kernels sequentially
- running copy and compute (`mul_fp32_ilp*`) kernels colocated

Use the compute `mul_fp32_ilp*` kernel to increase the pressure on the warp scheduler. By using a higher ILP degree or more warps, you can increase the kernel's IPC. Despite complementary resource profiles, both kernels can interfere at the warp scheduler level.

**Note**: Current parameters in the `run_ipc_experiment.sh` scripts are tailored for the H100. For other GPUs adjust values as follows:
- `threads_per_tb_comp` can be used next to ILP as a measure to modify the compute kernel ILP.
- `num_itrs_comp` and `num_itrs_copy` should be chosen such that both kernels have similar runtimes when run alone. Note this may change based on the chosen ILP degree.
- `num_bytes` Adjust on based on your GPU to avoid running out of memory.
- `ILP` should be set depending on the ILP degree you want to use in the compute kernel.

### Relevant NCU metrics
The relevant NCU metrics for this experiment are:
1. Average issued instructions per cycle per SM while active [instr/cycle]
    - `sm__inst_issued.avg.per_cycle_active`
2. Average issue slots busy [%]
    - `sm__inst_issued.avg.pct_of_peak_sustained_active`

