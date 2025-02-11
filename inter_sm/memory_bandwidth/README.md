# Memory Bandwidth interference
This folder contains the experimental code for section 4.1.3 of our [report](https://arxiv.org/pdf/2501.16909) demonstrating how colocation of two kernels can lead to memory bandwidth interference.

The main script can be run as follows:
```bash
./mem_bw <mode> <num_tb> <num_threads_per_tb> <itrs> <num_bytes>
```
#### Parameters
- `mode`: Execution mode for benchmark (0: profile, 0: run alone, 2: run sequential, 3: run collocated)
- `num_tb`: Number of thread blocks for copy kernel
- `num_threads_per_tb`: Number of threads per block for copy kernel
- `itrs`: Number of iterations for copy kernel
- `num_bytes`: Number of bytes to copy be copy kernel

### Running paper experiment
#### Measuring achieved maximum memory bandwidth
First, measure the actual achieved memory bandwidth rather than using the theoretical memory bandwidth:
```bash
./mem_bw 1 264 1024 1000 4294967296
```
Configuration for H100:
- 264 thread blocks (2 per SM)
- 1024 threads per block
- 2048 total threads per SM (H100 maximum)
- 4GB input array

Choose `num_tb` and `num_threads_per_tb` to maximize number of threads per SM.

#### Measuring interference
```bash
sh run_membw_experiment.sh
```
The script
- Gradually increases thread blocks per kernel, and for each iteration
- Compares copy kernel latency: alone vs. colocated
- Uses separate processes for each kernel and restricts each to 50% of SMs with MPS. This eliminates intra-SM interference in colocated scenario. 

**Note**: Current parameters in the `run_membw_experiment.sh` script are tailored to the H100 GPU. For other GPUs adjust the values as follows:
- `num_threads_per_tb`: should be set to `max_threads_per_multiprocessor / 2`.
- `num_bytes`: should be updated based on your memory capacity. It should exceed the L2 cache size and avoid out of memory errors.
- `num_tb`: adjust the loop based on the number of SMs present on your GPU. Be careful about incomplete thread block waves that can lead to misleading latencies and bandwidth. Example of the RTX3090 (82 SMs) with MPS (see also paper footnote):
    - available SMs with 50% MPS: 40 SMs (not 41)
    - maximum threads per SM: 1536
    - configuration: 82 thread blocks x 768 threads
    - Result: Incomplete wave causing idle SMs
        - Wave 1: 80 blocks (2 per SM, fully utilized)
        - Wave 2: 2 blocks (partially utilized, may lead to misleading metrics)

### Relevant NCU metrics
For this experiment you can track the following relevant NCU metrics:
1. Memory throughput [GB/s]
    - dram__bytes.sum.per_second
2. DRAM Throughput [%]
    - `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed`
    - note this % is relative to the theoretical maximum bandwidth, which is different from the maximum bandwidth you will most likely achieve

