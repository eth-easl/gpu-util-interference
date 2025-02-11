#!/bin/bash

# NOTE: current arguments are tailored to the H100
# adapt them based on the GPU you are running on
# don't forget to set the BUILD_DIR env variable

num_tb=132 # TODO: set to number of SMs present on GPU
num_threads_per_tb=1024 # TODO: set to max_threads_per_sm / 2
num_itrs=100000

# Measuring latnecy of single sleep kernel
$BUILD_DIR/tb_scheduler 1 $num_tb $num_threads_per_tb $num_itrs

# Measuring sequential latency of two sleep kernels
$BUILD_DIR/tb_scheduler 2 $num_tb $num_threads_per_tb $num_itrs

# Measuring colocated latency of two sleep kernels
$BUILD_DIR/tb_scheduler 3 $num_tb $num_threads_per_tb $num_itrs

# DOUBLING NUMBER OF THREAD BLOCKS

# Measuring latnecy of single sleep kernel with double the number of thread blocks
$BUILD_DIR/tb_scheduler 1 $((num_tb * 2)) $num_threads_per_tb $num_itrs

# Measuring sequential latency of two sleep kernels with double the number of thread blocks
$BUILD_DIR/tb_scheduler 2 $((num_tb * 2)) $num_threads_per_tb $num_itrs

# Measuring colocated latency of two sleep kernels with double the number of thread blocks
$BUILD_DIR/tb_scheduler 3 $((num_tb * 2)) $num_threads_per_tb $num_itrs