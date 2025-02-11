#include <iostream>
#include "kernels.cuh"
#include "cuda_helper.cuh"

using namespace std;

int main(int argc, char *argv[])
{
    if (argc < 5)
    {
        cout << "Please provide 4 arguments!" << endl;
        exit(1);
    }

    int mode = stoi(argv[1]);               // 0: profile, 1: run alone, 2: run sequential, 3: run colocated
    int num_tb = stoi(argv[2]);             // number of thread blocks
    int num_threads_per_tb = stoi(argv[3]); // number of threads per block
    long long num_itrs = stoll(argv[4]);    // number of iterations per kernel

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
    string launch_config = "Launch Config: (" + to_string(num_tb) + ", " + to_string(num_threads_per_tb) + ")";

    float *a1, *b1, *c1, *d_a1, *d_b1, *d_c1;
    float *a2, *b2, *c2, *d_a2, *d_b2, *d_c2;

    init_compute_memory<float>(&a1, &b1, &c1, &d_a1, &d_b1, &d_c1, num_threads_per_tb);
    auto args1 = std::make_tuple(d_a1, d_b1, d_c1, num_itrs);

    auto kernel = mul_fp32_ilp4;

    if (mode == 0)
    {
        cout << "----------------------------------------" << endl;
        cout << "Measuring GPU utilization (NVML) of single compute kernel - " << launch_config << endl;
        run_kernel_alone_with_nvml(kernel, gridDim, blockDim, args1, num_runs);
    }
    else if (mode == 1)
    {
        cout << "----------------------------------------" << endl;
        cout << "Measuring latency of single compute kernel - " << launch_config << endl;
        run_kernel_alone(kernel, gridDim, blockDim, args1, num_runs);
    }
    else if (mode == 2 || mode == 3)
    {
        // init memory and arguments for second kernel
        init_compute_memory<float>(&a2, &b2, &c2, &d_a2, &d_b2, &d_c2, num_threads_per_tb);
        auto args2 = std::make_tuple(d_a2, d_b2, d_c2, num_itrs);

        if (mode == 2)
        {
            cout << "----------------------------------------" << endl;
            cout << "Measuring sequential latency of two compute kernels - " << launch_config << endl;
            run_kernels_sequential(kernel, kernel, gridDim, blockDim, args1, gridDim, blockDim, args2, num_runs);
        }
        else
        {
            cout << "----------------------------------------" << endl;
            cout << "Measuring colocated latency of two compute kernels - " << launch_config << endl;
            run_kernels_colocated(kernel, kernel, gridDim, blockDim, args1, gridDim, blockDim, args2, num_runs);
        }

        free_compute_memory<float>(&a2, &b2, &c2, &d_a2, &d_b2, &d_c2);
    }
    else
    {
        cerr << "Invalid mode!" << endl;
        exit(1);
    }

    free_compute_memory<float>(&a1, &b1, &c1, &d_a1, &d_b1, &d_c1);
}