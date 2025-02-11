# Thread Block Scheduler Interference
This folder contains the experimental code for section 4.1.1 of our [report](https://arxiv.org/pdf/2501.16909) demonstrating how thread block scheduling becomes sequential when exceeding per-SM thread limits.

The main script can be run as follows
```bash
./tb_scheduler <mode> <num_tb> <num_threads_per_tb> <num_itrs>
```
#### Parameters
- `mode`: Execution mode for benchmark (0: profile, 1: run alone, 2: run sequential, 3: run collocated)
- `num_tb`: Number of thread blocks for the sleep kernel
- `num_threads_per_tb`: Number of threads per block per sleep kernel
- `num_itrs`: Number of iterations per kernel

### Running the paper experiment
```bash
sh run_tb_experiment.sh
```

The script will perform two runs:
- **Run 1 - Concurrent Execution**: Each kernel runs one thread block per SM and the blocks of both kernels execute concurrently. The colocated runtime is 2x faster than the sequential runtime.
- **Run 2 - Sequential Execution**: By doubling the amount of thread blocks per kernel, each kernel will run two thread blocks per SM. Each kernel independently saturates maximum SM thread count. The thread block scheduler will serialize the kernel launches -> colocated runtime matches sequential runtime.

**Note**: Current parameters in the script are tailored to the H100 GPU. For other GPUs adjust the values as follows:
- `num_tb` should be set to the number of SMs present on the GPU.
- `num_threads_per_tb` should be set to `max_threads_per_multiprocessor / 2`.

### Relevant NCU metrics
The `Occupancy` section in the NCU report contains information on how many thread blocks of a given kernel can run concurrently on the same SM without exceeding the SMs resources. Note that these limits apply to blocks from the same kernel. Limits may differ when mixing blocks from different kernels.
1. Max blocks allowed based on used registers per thread
    - `launch__occupancy_limit_registers`
2. Max blocks allowed based on shared memory usage per thread block
    - `launch__occupancy_limit_shared_mem`
3. Max blocks allowed based on warp count per SM
    - `launch__occupancy_limit_warps`
4. Hardware limit on concurrent blocks per SM
    - `launch__occupancy_limit_blocks`



