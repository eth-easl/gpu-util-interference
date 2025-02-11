# L1 cache interference
This folder contains the experimental code support for section 4.2.1 of our [report](https://arxiv.org/pdf/2501.16909) demonstrating how kernel colocation can lead to L1 cache interference.

The main script can be run as follows:
```bash
./l1_cache <mode> <num_threads_per_tb> <num_bytes_per_tb> <unified_l1_cache_size> <num_itrs>
```
#### Parameters
- `mode`: Execution mode for benchmark (0: profile, 0: run alone, 2: run sequential, 3: run collocated)
- `num_threads_per_tb`: Number of threads per thread block for copy kernel
- `num_bytes_per_tb`: Number of bytes to copy per thread block
- `unified_l1_cache_size`: Size of unified L1 cache (including shared memory)
- `num_itrs`: Number of iterations for copy kernel

### Running paper experiment
```bash
sh run_l1_experiment.sh
```
The script
- Gradually increases data copy size per thread block
- For each size, it compares the colocated latency of two `copy_kernel_per_tb` kernels against running them sequentially

**Note**: Current parameters in the `run_l1_experiment.sh` scripts are tailored for the H100. For other GPUs adjust values as follows:
- `num_threads_per_tb` should be set to the `num_warp_schedulers_per_sm * 32 / 2`
    - Each kernel's thread blocks should occupy half SM subpartitions
    - This should help to eliminate interference within an SM subpartition and isolate L1 cache interference
    - Note: This is no guarantee that two different kernels will not end up sharin the same SMSP, but we believe this will take place.
- `unififed_l1_cache_size` should be set to the unfied L1 cache size (including shared memory) of your GPU.
    - We use this value to align memory regions and prevent memory regions of different thread blocks to overlap.
- `num_bytes_per_tb` should be adjusted to your unified L1 cache size.
    - Start small and incrementally increase
    - You want to identify the threshold where interference starts to occur

### Relevant NCU metrics
For this experiment, you can track the L1 cache hit rate of the kernel in isolation. A kernel which already has a low L1 hit rate when run alone, is less likely to suffer from L1 cache interference introduced through colocation:
1. L1/TEX cache hit rate:
    - `l1tex__t_sector_hit_rate.pct`
