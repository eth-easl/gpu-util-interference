add_library(gpu_util_bench kernels.cu)

target_include_directories(gpu_util_bench PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(gpu_util_bench PUBLIC nvidia-ml)