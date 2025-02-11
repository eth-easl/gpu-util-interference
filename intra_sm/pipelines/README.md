# Pipeline Interference
This folder contains the experimental code support for section 4.2.3 of our [report](https://arxiv.org/pdf/2501.16909).  The benchmark demonstrates how compute pipelines (e.g., FP64, FMA) can be subject to interference during kernel colocation.

The main script can be run as follows:
```bash
./pipelines <mode> <ilp_degree> <num_threads_per_tb> <num_itrs>
```
#### Parameters
- `mode`: Execution mode for benchmark (0: profile, 1: run alone, 2: run sequential, 3: run collocated)
- `ilp_degree`: Instruction-level parallelism degree of compute kernel
- `num_threads_per_tb`: Number of threads per thread block for compute kernel
- `num_itrs`: Number of iterations for compute kernel

### Running paper experiment
```bash
sh run_pipeline_experiment.sh
```
The script
- Iteratively adjusts the ILP degree of the compute kernel (`mul_fp64_ilp*`)
- For each iteration, it compares
    - latency of two compute kernels when run sequentially
    - latency of two compute kernels when colocated using CUDA streams
- Note: ILP degree increases IPC without requiring more warps per thread block

**Note**: Current parameters in the `run_pipeline_experiment.sh` scripts are tailored for the H100. For other GPUs adjust values as follows:
- `num_threads_per_tb` should be set to `num_warp_schedulers_per_sm * 32` to ensure one warp per scheduler

#### Choice of using a FP64 kernel
We chose the `mul_fp64_ilp*` kernel because on H100:
- FP64 pipeline saturates at IPC of 2
- Demonstrates pipeline interference before warp scheduler (IPC of 4) saturates

Saturating a computation pipeline is frequently associated with a high instructions per cycle (IPC), as this necessitates issuing a new warp instruction (comprising 32 threads) to the pipeline in every cycle. However, on the H100 architecture, the peak performance for FP64 operations is only half that of FP32 operations. Consequently, a new warp instruction can be issued to the FP64 pipeline, on average, every two cycles, whereas the FP32 pipeline can receive a new warp instruction every cycle on average.

As the [Nsight Compute notes](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#id27) state the FP64 pipeline varies a lot across architectures and you may need to use a different kernel for your GPU. Consider the [throughput numbers](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#arithmetic-instructions) for different data types for your GPU.

### Relevant NCU metrics
For this experiment the relevant NCU metrics are:
1. Pipe utilization (% of peak instructions executed)
    - `sm__inst_executed_<pipeline>_fma.avg.pct_of_peak_sustained_active`
2. Average issued instructions per cycle per SM while active [instr/cycle]
    - `sm__inst_issued.avg.per_cycle_active`
3. Average issue slots busy [%]
    - `sm__inst_issued.avg.pct_of_peak_sustained_active`