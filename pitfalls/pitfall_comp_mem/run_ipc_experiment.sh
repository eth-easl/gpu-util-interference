#!/bin/bash

# NOTE: current arguments are tailored to the H100
# adapt them based on the GPU you are running on
# don't forget to set the BUILD_DIR env variable

# ensure total number of threads of both kernels does not exceed
# prevent colocation on same SM based on max number of threads per SM
NUM_THREADS_PER_BLOCK=1024

# TODO: ensure similar runtimes for both kernels when run in isolation
NUM_ITERS_COMP_BENCH=30000000
NUM_ITERS_MEM_BENCH=180

# measure latency of compute and copy kernel when run in isolation
./$BUILD_DIR/comp_mem_ipc 1 $NUM_THREADS_PER_BLOCK $NUM_ITERS_COMP_BENCH $NUM_ITERS_MEM_BENCH

# masure latency of compute and copy kernel when running concurrently
./$BUILD_DIR/comp_mem_ipc 3 $NUM_THREADS_PER_BLOCK $NUM_ITERS_COMP_BENCH $NUM_ITERS_MEM_BENCH