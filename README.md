# Measuring GPU utilization one level deeper
This repository contains the supporting code for our paper [Measuring GPU Utilization one level deeper](https://arxiv.org/pdf/2501.16909). We present a comprehensive suite of CUDA benchmarks designed to identify and measure interference across various GPU resources.

## Repository Structure
The codebase is organized into the following primary directories:
- `gpu_util_bench_lib/`: A shared library containing CUDA kernels and helper functions for kernel launching
- `inter_sm/`: Benchmarks for measuring interference and utilization across Streaming Multiprocessors (SM) (Paper section 4.1)
- `intra_sm/`: Benchmarks for measuring interference and utilization within SMs (Paper section 4.2)
- `mm_pytroch/`: Example demonstrating interference patterns on production ML kernels (Paper section 4.3)
- `pitfalls/`: Examples illustrating common limitations in current interference prediction approaches (Paper section 3)

## Requirements and Installation
### Prerequisites
To benchmarks require the follwoing dependencies:
- [CMake](https://cmake.org/download/) (version >= 3.22)
- C++17 or later
- [CUDA toolkit](https://developer.nvidia.com/cuda-downloads) (validated with CUDA 12.5 and 12.6)
- NVIDIA GPU driver (can be installed alongside CUDA toolkit)

Note: Our benchmarks currently do not support AMD GPUs.

### Compilation Instructions
1. Determine your GPU's Compute Capability using [nvidia-smi](https://docs.nvidia.com/deploy/nvidia-smi/index.html):
    ```bash
    nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1
    ```

2. Update the Compute Capability in `CMakeLists.txt`:
    ```bash
    set(CMAKE_CUDA_ARCHITECTURES 90)  # Modify based on your GPU
    ```

3. Build the repository
    ```bash
    mkdir build && cd build
    cmake ..
    cmake --build .
    ```

## Running experiments
### Benchmark Execution
Each directory contains detailed instructions for executing the benchmarks and reproducing paper experiments. The provided scripts are optimized for the H100 GPU. Users with different GPU architectures may need to adjust script parameters accordingly.

**Important**: Before running experiments, set the `BUILD_DIR` environment variable to match your build directory.
```bash
export BUILD_DIR=$HOME/gpu-util-interference/build # update based on location of your build directory
```

### Performance Analysis
#### NCU Metrics Collection
To gather detailed performance metrics for isolated kernel execution, use the [Nsight Compute Profiler](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html). When profiling with NCU, specify `mode=0` in the scripts:
```bash
ncu -f -o ncu.ncu-rep --set full <executable>
```

#### CUDA Trace Collection
For analyzing kernel co-location scenarios, we recommend collecting CUDA traces using the [Nsight Systems Profiler](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) to visualize kernel overlap patterns and verify concurrent execution.
```bash
nsys profile --force-overwrite true -o nsys.nsys-rep --trace cuda <executable>
```

## Experimental Setup
Our paper's results were obtained using the following hardware configurations:
- [H100 NVL](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/h100/PB-11773-001_v01.pdf):
    - CUDA version 12.5
    - GPU driver version 555.42.06
    - Nsight Compute version 2024.2.1.0
    - Nsight Systems version 2024.2.3.38
- [GeForce RTX3090](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf):
    - CUDA version 12.6
    - GPU driver version 560.35.03
    - Nsight Compute version 2024.3.1.0
    - Nsight Systems version 2024.4.2.133

## Paper
If you use our benchmarks, please cite our paper:
```bibtex
@article{elvinger2025measuring,
  title={Measuring GPU utilization one level deeper},
  author={Elvinger, Paul and Strati, Foteini and Jerger, Natalie Enright and Klimovic, Ana},
  journal={arXiv preprint arXiv:2501.16909},
  year={2025}
}
```