# L2 Cache Interference
This folder contains the experimental code support for section 4.1.2 of our [report](https://arxiv.org/pdf/2501.16909) demonstrating how kernel colocation can lead to L2 cache interference.

The main script can be run as follows:
```bash
./l2_cache <mode> <num_tb> <num_threads_per_tb> <itrs> <num_bytes>
```
#### Parameters
- `mode`: Execution mode for benchmark (0: profile, 0: run alone, 2: run sequential, 3: run collocated)
- `num_tb`: Number of thread blocks for copy kernel
- `num_threads_per_tb`: Number of threads per block for copy kernel
- `itrs`: Number of iterations for copy kernel
- `num_bytes`: Number of bytes to copy be copy kernel

### Running paper experiments
```bash
sh run_l2_experiment.sh
```
The script
- Gradually increases the number of bytes to be copied by the copy kernel
- For each iteration it compares the latency of the copy kernel when colocated with CUDA streams compared to running alone

**Note**: Current parameters in the `run_l2_experiment.sh` scripts are tailored for the H100. For other GPUs adjust values as follows:
- `num_tb` should be set to half the number of SMs on your GPU. This should enable separate SM allocation for each copy kernel instance, eliminating intra-SM interference during colocation.
- `num_bytes` should be adjusted to your L2 cache size. Start small and interatively increase the size in order to identify the threshold where interference begins.

### Relevant NCU metrics
For this experiment, you can track the L2 cache hit rate of the kernel in isolation. A kernel which already has a low L2 hit rate when run alone, is less likely to suffer from L2 cache interference introduced through colocation:
1. L2 cache hit rate:
    - lts__t_sector_hit_rate.pct