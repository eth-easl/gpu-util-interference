#!/bin/bash

# NOTE: current arguments are tailored to the H100
# adapt them based on the GPU you are running on
# don't forget to set the BUILD_DIR env variable

num_threads_per_tb=128 # run 1 warp per SM subpartition
num_itrs=20000000


for ILP in {1..4..1}; do
    # run two compute kernels sequentially
    $BUILD_DIR/pipelines 2 $ILP $num_threads_per_tb $num_itrs

    # run two compute kernels colocated using CUDA streams
    $BUILD_DIR/pipelines 3 $ILP $num_threads_per_tb $num_itrs
done