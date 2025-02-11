# Pitfall: Using nvidia-smi/NVML to measure GPU utilization
This benchmark demonstrates why nvidia-smi/NVML is not well suited for measuring GPU utilization. Our experiments show that these tools may not accurately reflect the actual GPU resource usage of CUDA kernels.

The main script can be run as
```bash
./nvml_util <mode> <num_tb> <num_threads_per_tb> <num_itrs>
```
#### Parameters
- `mode`: Execution mode for benchmark (0: profiling, 1: run alone, 2: run sequential, 3: run collocated)
- `num_tb`: Number of thread blocks for the compute kernel
- `num_threads_per_tb`: number of threads per block for the compute kernel
- `num_itrs`: number of iterations for the compute kernel

### Running the paper experiment
```bash
sh run_util_experiment.sh
```

The benchmark reveals an important limitation of nvidia-smi/NVML tools:
- When running a kernel with just one thread block, nvidia-smi/NVML reports GPU utilization at or near 100%
- However, running two instances of the same compute kernel simultaneously (each with one thread block) takes the same time as running a single instance
- This shows the GPU actually has spare computational capacity, despite nvidia-smi/NVML reporting full utilization