#!/bin/bash

# NOTE: current arguments are tailored to the H100
# adapt them based on the GPU you are running on
# don't forget to set the BUILD_DIR env variable

num_threads_per_tb=64 # should be set to num_schedulers_per_sm * 32 / 2
unified_l1_cache_size=256 # TODO: update based on your GPU
num_itrs=15000

# TODO: update for loop based on your GPUs unified L1 cache size
# for (num_bytes_per_tb = 32KB; num_bytes_per_tb <= 128KB; num_bytes_per_tb += 32KB)
for num_bytes_per_tb in {32768..131072..32768}; do
    # measure sequential latency of two copy kernel
    ./$BUILD_DIR/l1_cache 2 $num_threads_per_tb $num_bytes_per_tb $unified_l1_cache_size $num_itrs

    # measure concurrent latency of two copy kernels
    ./$BUILD_DIR/l1_cache 3 $num_threads_per_tb $num_bytes_per_tb $unified_l1_cache_size $num_itrs
done;