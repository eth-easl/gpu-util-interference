#!/bin/bash

# NOTE: current arguments are tailored to the H100
# adapt them based on the GPU you are running on
# don't forget to set the BUILD_DIR env variable

num_tb=66 # TODO: set to half num SM on the GPU
num_threads_per_tb=1024
num_itrs=10000

# TODO: update for loop based on your GPUs L2 cache size
# for (NUM_BYTES = 8MB; NUM_BYTES <= 40MB; NUM_BYTES += 8MB)
for NUM_BYTES in {8388608..41943040..8388608}; do
    # Measure latency of single copy kernel
    ./$BUILD_DIR/l2_cache 1 $num_tb $num_threads_per_tb $num_itrs $NUM_BYTES

    # Measure latency of two colocated copy kernel
    ./$BUILD_DIR/l2_cache 3 $num_tb $num_threads_per_tb $num_itrs $NUM_BYTES
done;
