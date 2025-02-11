#include <iostream>
#include "kernels.cuh"
#include "cuda_helper.cuh"

int main(int argc, char **argv)
{
    if (argc < 6)
    {
        std::cout << "Provide 5 arguments" << std::endl;
        return 1;
    }

    int device_id = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    int num_sms = prop.multiProcessorCount;
    if (!prop.concurrentKernels)
    {
        cerr << "Concurrent Kernels not supported on this device." << endl;
        return 1;
    }

    int mode = stoi(argv[1]);                  // 0: profile, 1: run alone, 2: run sequential, 3: run colocated
    int num_threads_per_tb = stoi(argv[2]);    // number of threads per block
    int num_bytes_per_tb = stoi(argv[3]);      // number of bytes to copy per thread block
    int unified_l1_cache_size = stoi(argv[4]); // size of unified L1 cache in bytes KB
    long long num_itrs = stoll(argv[5]);       // number of iterations for copy kernel

    int num_runs = (mode == 0) ? 1 : 10;                // in profiling mode, perform only single run
    long long num_floats_per_tb = num_bytes_per_tb / 4; // divide by 4 bytes per float
    int unified_l1_cache_size_bytes = unified_l1_cache_size * 1024;

    // allocate separate non overlapping memory regions for each thread block
    // the size of each region is equal to the unified L1 cache size
    // if num_bytes_per_tb > unified_l1_cache_size, then we need to allocate multiple regions per thread block
    int num_regions_per_block = (num_bytes_per_tb + unified_l1_cache_size_bytes - 1) / unified_l1_cache_size_bytes;
    int tot_elems = num_sms * num_regions_per_block * unified_l1_cache_size_bytes / 4; // divide by 4 bytes per float

    // set cache preference to default for the copy_kernel_per_tb CUDA kernel
    cudaFuncSetAttribute(copy_kernel_per_tb, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutDefault);

    dim3 gridDim(num_sms);
    dim3 blockDim(num_threads_per_tb);
    string launch_config = "Launch Config: (" + to_string(num_sms) + ", " + to_string(num_threads_per_tb) + ") - copying " + to_string(num_bytes_per_tb / 1024) + " KB per thread block";

    float *in1, *out1, *d_in1, *d_out1;
    float *in2, *out2, *d_in2, *d_out2;

    init_copy_memory<float>(&in1, &out1, &d_in1, &d_out1, tot_elems);
    auto kernel_args1 = std::make_tuple(d_in1, d_out1, num_floats_per_tb, num_itrs, unified_l1_cache_size_bytes);

    if (mode == 0 || mode == 1)
    {
        cout << "------------------------------------" << endl;
        cout << "Running copy kernel alone - " << launch_config << endl;
        run_kernel_alone(copy_kernel_per_tb, gridDim, blockDim, kernel_args1, num_runs);
    }
    else if (mode == 2 || mode == 3)
    {
        init_copy_memory<float>(&in2, &out2, &d_in2, &d_out2, tot_elems);
        auto kernel_args2 = std::make_tuple(d_in2, d_out2, num_floats_per_tb, num_itrs, unified_l1_cache_size_bytes);

        if (mode == 2)
        {
            cout << "------------------------------------" << endl;
            cout << "Running two copy kernels sequentially - " << launch_config << endl;
            run_kernels_sequential(copy_kernel_per_tb, copy_kernel_per_tb, gridDim, blockDim, kernel_args1, gridDim, blockDim, kernel_args2, num_runs);
        }
        else
        {
            cout << "------------------------------------" << endl;
            cout << "Running two copy kernels colocated - " << launch_config << endl;
            run_kernels_colocated(copy_kernel_per_tb, copy_kernel_per_tb, gridDim, blockDim, kernel_args1, gridDim, blockDim, kernel_args2, num_runs);
        }

        free_copy_memory<float>(&in2, &out2, &d_in2, &d_out2);
    }

    free_copy_memory<float>(&in1, &out1, &d_in1, &d_out1);

    return 0;
}