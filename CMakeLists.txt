cmake_minimum_required(VERSION 3.22)
project(gpu_util_benchmarks LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

# Set CUDA architectures, 86 for RTX 3090, 90 for H100
set(CMAKE_CUDA_ARCHITECTURES 90)

find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})

add_subdirectory(gpu_util_bench_lib)
add_subdirectory(mm_pytorch)

# create an executable for each benchmark

# pitfalls
add_executable(nvml_util ${CMAKE_CURRENT_SOURCE_DIR}/pitfalls/pitfall_nvidia_smi/nvml_util.cu)
add_executable(achieved_occupancy ${CMAKE_CURRENT_SOURCE_DIR}/pitfalls/pitfall_achieved_occupancy/achieved_occupancy.cu)
add_executable(comp_mem_ipc ${CMAKE_CURRENT_SOURCE_DIR}/pitfalls/pitfall_comp_mem/comp_mem_ipc.cu)

# inter SM
add_executable(l2_cache ${CMAKE_CURRENT_SOURCE_DIR}/inter_sm/l2_cache/l2_cache.cu)
add_executable(mem_bw ${CMAKE_CURRENT_SOURCE_DIR}/inter_sm/memory_bandwidth/mem_bw.cu)
add_executable(tb_scheduler ${CMAKE_CURRENT_SOURCE_DIR}/inter_sm/thread_block_scheduler/tb_scheduler.cu)

# intra SM
add_executable(ipc ${CMAKE_CURRENT_SOURCE_DIR}/intra_sm/ipc/ipc.cu)
add_executable(pipelines ${CMAKE_CURRENT_SOURCE_DIR}/intra_sm/pipelines/pipelines.cu)
add_executable(l1_cache ${CMAKE_CURRENT_SOURCE_DIR}/intra_sm/l1_cache/l1_cache.cu)

# link all executables to the gpu_util_bench library
target_link_libraries(achieved_occupancy PUBLIC gpu_util_bench)
target_link_libraries(comp_mem_ipc PUBLIC gpu_util_bench)
target_link_libraries(nvml_util PUBLIC gpu_util_bench)

target_link_libraries(l2_cache PUBLIC gpu_util_bench)
target_link_libraries(mem_bw PUBLIC gpu_util_bench)
target_link_libraries(tb_scheduler PUBLIC gpu_util_bench)

target_link_libraries(ipc PUBLIC gpu_util_bench)
target_link_libraries(pipelines PUBLIC gpu_util_bench)
target_link_libraries(l1_cache PUBLIC gpu_util_bench)
