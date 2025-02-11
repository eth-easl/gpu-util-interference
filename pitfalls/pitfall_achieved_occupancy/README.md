# Pitfall: Using achieved occupancy as a kernel's compute requirements
This folder contains the experimental code supporting Pitfall 2 in Section 3 of our [report](https://arxiv.org/pdf/2501.16909). The benchmark demonstrates why `Achieved Occupancy` may be an inadequate metric for estimating a kernel's compute requirements. We test this by colocating two compute (`mul_fp32_ilp4`) kernels using both CUDA streams and MPS

The main script can be run as follows:
```bash
./achieved_occupancy <mode> <num_tb> <num_threads_per_tb> <num_itrs>
```
#### Parameters
- `mode`: Execution mode for benchmark (0: profiling, 1: run alone, 2: run sequential, 3: run collocated)
- `num_tb`: Number of thread blocks for the compute kernel
- `num_threads_per_tb`: number of threads per block for the compute kernel
- `num_itrs`: number of iterations for the compute kernel

### Running the paper experiments
**Note**: Current parameters in the `run_mps_experiment.sh` and `run_stream_experiment.sh` scripts are tailored for the H100. For other GPUs adjust values as follows:
- `num_tb` should be set to the number of SMs present on the GPU
- `achieved_occupancy` should be set to the achieved occupancy (`sm__warps_active.avg.pct_of_peak_sustained_active`) from the NCU report when profiling the compute kernel in isolation (see below).

#### Profile Achieved Occupancy with NCU
First, measure the compute kernel's achieved occupancy using NCU:
```bash
ncu -f -o achieved_occupancy.ncu-rep --set full ./achieved_occupancy 0 132 128 1000000
```
- launch one thread block per SM (132 SMs for H100) and 1 warp per subpartition (4 SMSP)
- check the `sm__warps_active.avg.pct_of_peak_sustained_active` metric in the report

#### Colocate kernels with CUDA streams on all SMs
```bash
sh run_stream_experiment.sh
```
The experiment compares
- Individual latency of one compute kernel
- Sequential latency of two compute kernels
- Makespan when two compute kernels run colocated using CUDA streams

The experiment demonstrates that achieved occupancy is not an appropriate colocation metric, as it doesn't accurately reflect how well a small number of warps can saturate SM components (warp scheduler, FMA pipeline)


#### Colocate kernels with MPS on separate sets of SMs
```bash
sh run_mps_experiment.sh
```
The experiment compares
- Individual latency of one compute kernel on all SMs without MPS
- Sequential latency of two compute kernels an all SMs without MPS
- Makespan when two compute kernels run colocated on separate SMs with MPS. Each kernel is launched from a different process and the process is restricted to a portion of the SMs, by setting `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE = achieved_occupancy`.

The experiment demonstrates that achieved occupancy poorly indicates the number of SMs needed to run a kernel efficiently.


### Relevant NCU metrics
By using a high level of Instruction Level Parallelism (ILP) for the `mul_fp32_ilp4` kernel, the kernel is able to issue a high number of instructions per cycle (IPC) despite using only 4 warps per SM. The consequence is a high FMA pipeline utilization. The relevant NCU metrics for this experiment are:
1. Achieved Occupancy [%]
    - `sm__warps_active.avg.pct_of_peak_sustained_active`
2. Average issued instructions per cycle per SM while active [instr/cycle]
    - `sm__inst_issued.avg.per_cycle_active`
3. Average issue slots busy [%]
    - `sm__inst_issued.avg.pct_of_peak_sustained_active`
4. FMA pipeline utilization [% of peak instructions executed]
    - `sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active`
