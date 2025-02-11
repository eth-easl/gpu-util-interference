#!/bin/bash

# NOTE: current arguments are tailored to the H100
# adapt them based on the GPU you are running on
# don't forget to set the BUILD_DIR env variable

num_tb=132 # TODO: set to number of SMs present on GPU
num_threads_per_tb=128 # 4 warps
num_itrs=100000000
achieved_occupancy=6.25 # TODO: set to achieved occupancy of compute kernel from NCU report

echo "----------------------------------------"
echo "Running without MPS"

# measure latency of single compute kernel on all SMs WITHOUT MPS
./$BUILD_DIR/achieved_occupancy 1 $num_tb $num_threads_per_tb $num_itrs

# measure sequential latency of two compute kernels on all SMs WITHOUT MPS
./$BUILD_DIR/achieved_occupancy 2 $num_tb $num_threads_per_tb $num_itrs

echo "----------------------------------------"
echo "Starting MPS"
export CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps # Select a location that’s accessible to the given $UID
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log # Select a location that’s accessible to the given $UID
nvidia-cuda-mps-control -d # Start the daemon.

function cleanup() {
    echo "Shutting down MPS control daemon..."
    echo quit | nvidia-cuda-mps-control
    echo "MPS control daemon shut down."
}

# always shut down MPS control daemon
trap "cleanup; exit" SIGINT

echo "Measuring latency of two colocated compute kernels WITH MPS. Each kernel uses $achieved_occupancy % of the SMs"
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$achieved_occupancy ./$BUILD_DIR/achieved_occupancy 1 $num_tb $num_threads_per_tb $num_itrs &
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$achieved_occupancy ./$BUILD_DIR/achieved_occupancy 1 $num_tb $num_threads_per_tb $num_itrs

cleanup
