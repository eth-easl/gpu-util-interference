#!/bin/bash

# NOTE: current arguments are tailored to the H100
# adapt them based on the GPU you are running on
# don't forget to set the BUILD_DIR env variable

NUM_THREADS_PER_BLOCK=1024 # TODO: set to max_threads_per_sm / 2
NUM_ITRS=50
NUM_BYTES=4294967296 # 4 GB, TODO: update based on your GPU memory

echo "------------------------------------"
echo "Starting MPS"
export CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps # Select a location that’s accessible to the given $UID
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log # Select a location that’s accessible to the given $UID
nvidia-cuda-mps-control -d

function cleanup() {
    echo "Shutting down MPS control daemon..."
    echo quit | nvidia-cuda-mps-control
    echo "MPS control daemon shut down."
}

# always shut down MPS control daemon
trap "cleanup; exit" SIGINT

# TODO: adjust the loop based on the number of SMs on your GPU
for NUM_TB in {33..132..33}; do
    # measuring latency of a single copy kernel on 50% of SMs
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 ./$BUILD_DIR/mem_bw 1 $NUM_TB $NUM_THREADS_PER_BLOCK $NUM_ITRS $NUM_BYTES

    # measuring latencies of two copy kernels colocated each running on 50% of SMs
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 ./$BUILD_DIR/mem_bw 1 $NUM_TB $NUM_THREADS_PER_BLOCK $NUM_ITRS $NUM_BYTES &
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 ./$BUILD_DIR/mem_bw 1 $NUM_TB $NUM_THREADS_PER_BLOCK $NUM_ITRS $NUM_BYTES
done

cleanup