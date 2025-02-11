#!/bin/bash

# NOTE: current arguments are tailored to the H100
# adapt them based on the GPU you are running on
# don't forget to set the BUILD_DIR env variable

threads_per_tb_copy=1024
threads_per_tb_comp=128 # TODO: update based on scenario you are running
num_itrs_comp=20000000 # TODO: update to match isolated copy runtime
num_itrs_copy=40 # TODO: update to match isolated compute runtime
num_bytes=4294967296 # 4 GB, update if necessary to avoid OOM errors

ILP=4 # TODO: update compute ilp level based on scenario you are running

# Measure the latency of a single copy kernel
$BUILD_DIR/ipc 1 0 $threads_per_tb_copy $threads_per_tb_comp $num_itrs_copy $num_itrs_comp $num_bytes

# Measure the latency of a single compute kernel
$BUILD_DIR/ipc 1 $ILP $threads_per_tb_copy $threads_per_tb_comp $num_itrs_copy $num_itrs_comp $num_bytes

# Measure the latency of a copy and compute kernel running sequentially
# NOTE: update num_itrs_comp and num_itrs_copy such that their isolated runtime is similar
$BUILD_DIR/ipc 2 $ILP $threads_per_tb_copy $threads_per_tb_comp $num_itrs_copy $num_itrs_comp $num_bytes

# Measure the latency of a copy and compute kernel running concurrently
$BUILD_DIR/ipc 3 $ILP $threads_per_tb_copy $threads_per_tb_comp $num_itrs_copy $num_itrs_comp $num_bytes