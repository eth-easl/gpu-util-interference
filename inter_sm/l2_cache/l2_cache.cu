#include <iostream>
#include "kernels.cuh"
#include "cuda_helper.cuh"

using namespace std;

int main(int argc, char *argv[])
{
    if (argc < 6)
    {
        cout << "Please provide 5 arguments!" << endl;
        exit(1);
    }

    int mode = stoi(argv[1]);               // 0: profile, 1: run alone, 2: run sequential, 3: run colocated
    int num_tb = stoi(argv[2]);             // number of thread blocks per kernel
    int num_threads_per_tb = stoi(argv[3]); // number of threads per block
    int itrs = stoi(argv[4]);               // number of iterations per kernel
    long long num_bytes = stol(argv[5]);    // number of bytes to copy

    size_t num_elems = num_bytes / sizeof(float); // number of floats to copy
    float num_mb = stol(argv[5]) / (1024 * 1024);

    int num_runs = (mode == 0) ? 1 : 10; // in profiling mode, perform only single run

    int device_id = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    if (!prop.concurrentKernels)
    {
        cerr << "Concurrent Kernels not supported on this device." << endl;
        return 1;
    }

    dim3 gridDim(num_tb);
    dim3 blockDim(num_threads_per_tb);
    string launch_config = "Launch Config: (" + to_string(num_tb) + ", " + to_string(num_threads_per_tb) + ") - copying " + to_string(num_mb) + " MB";

    auto kernel = copy_kernel;

    float *in1, *out1, *d_in1, *d_out1;
    float *in2, *out2, *d_in2, *d_out2;

    init_copy_memory<float>(&in1, &out1, &d_in1, &d_out1, num_elems);
    auto kernel_args1 = std::make_tuple(d_in1, d_out1, num_elems, itrs);

    if (mode == 0 || mode == 1)
    {
        cout << "----------------------------------------" << endl;
        cout << "Running copy kernel alone - " << launch_config << endl;
        run_kernel_alone(kernel, gridDim, blockDim, kernel_args1, num_runs);
    }
    else if (mode == 2 || mode == 3)
    {
        // initialize memory for second kernel
        init_copy_memory<float>(&in2, &out2, &d_in2, &d_out2, num_elems);
        auto kernel_args2 = std::make_tuple(d_in2, d_out2, num_elems, itrs);

        if (mode == 2)
        {
            cout << "----------------------------------------" << endl;
            cout << "Running copy kernel sequentially - " << launch_config << endl;
            run_kernels_sequential(kernel, kernel, gridDim, blockDim, kernel_args1, gridDim, blockDim, kernel_args2, num_runs);
        }
        else
        {
            cout << "----------------------------------------" << endl;
            cout << "Running copy kernel colocated - " << launch_config << endl;
            run_kernels_colocated(kernel, kernel, gridDim, blockDim, kernel_args1, gridDim, blockDim, kernel_args2, num_runs);
        }

        free_copy_memory<float>(&in2, &out2, &d_in2, &d_out2);
    }
    else
    {
        cerr << "Invalid mode!" << endl;
        exit(1);
    }
}