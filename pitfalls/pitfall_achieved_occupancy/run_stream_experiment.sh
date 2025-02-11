#!/bin/bash

# NOTE: current arguments are tailored to the H100
# adapt them based on the GPU you are running on
# don't forget to set the BUILD_DIR env variable

num_tb=132 # TODO: set to number of SMs present on GPU
num_threads_per_tb=128
num_itrs=100000000

# measure latency of single compute kernel on all SMs without MPS
./$BUILD_DIR/achieved_occupancy 1 $num_tb $num_threads_per_tb $num_itrs

# measure sequential latency of two compute kernels on all SMs without MPS
./$BUILD_DIR/achieved_occupancy 2 $num_tb $num_threads_per_tb $num_itrs

# measure colocated latency of two compute kernels with CUDA streams an all SMs without MPS
./$BUILD_DIR/achieved_occupancy 3 $num_tb $num_threads_per_tb $num_itrs
