#!/bin/bash

BUILD_DIR=../../build # TODO: update based on the location of your build folder

num_tb=1
num_threads_per_tb=1024
num_itrs=30000000
num_itrs_prof=40000000

# measure GPU utilization
./$BUILD_DIR/nvml_util 0 $num_tb $num_threads_per_tb $num_itrs_prof

# measure latency of single compute kernel
./$BUILD_DIR/nvml_util 1 $num_tb $num_threads_per_tb $num_itrs

# measure latency of two colocated comptue kernels
./$BUILD_DIR/nvml_util 3 $num_tb $num_threads_per_tb $num_itrs