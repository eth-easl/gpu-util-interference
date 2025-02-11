#include <iostream>
#include "kernels.cuh"
#include "cuda_helper.cuh"

using namespace std;

#define NUM_RUNS 10
#define NUM_PROF_RUNS 1

int main(int argc, char *argv[])
{

    if (argc < 5)
    {
        cerr << "Please provide 4 arguments!" << endl;
        exit(1);
    }

    int mode = stoi(argv[1]);               // 0: profile, 1: run alone, 2: run sequential, 3: run colocated
    int num_tb = stoi(argv[2]);             // number of thread blocks
    int num_threads_per_tb = stoi(argv[3]); // number of threads per block
    int num_itrs = stoi(argv[4]);           // number of iterations for the compute kernel

    int num_runs = (mode == 0) ? 1 : 10; // in profiling mode, perform only single run

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int num_sms = prop.multiProcessorCount;
    if (!prop.concurrentKernels)
    {
        cerr << "Concurrent Kernels not supported on this device." << endl;
        return 1;
    }

    cout << "Number of visible SMs: " << num_sms << endl;

    dim3 gridDim(num_tb);
    dim3 blockDim(num_threads_per_tb);
    string launch_config = "Launch Config: (" + to_string(num_tb) + ", " + to_string(num_threads_per_tb) + ")";

    // compute kernel used for experiment
    auto compute_kernel = mul_fp32_ilp4;

    float *a1, *b1, *c1, *d_a1, *d_b1, *d_c1;
    float *a2, *b2, *c2, *d_a2, *d_b2, *d_c2;

    init_compute_memory<float>(&a1, &b1, &c1, &d_a1, &d_b1, &d_c1, num_threads_per_tb);
    auto kernel_args1 = std::make_tuple(d_a1, d_b1, d_c1, num_itrs);

    if (mode == 0 || mode == 1)
    {
        cout << "----------------------------------------" << endl;
        cout << "Running compute kernel alone - " << launch_config << endl;
        run_kernel_alone(compute_kernel, gridDim, blockDim, kernel_args1, num_runs);
    }
    else if (mode == 2 || mode == 3)
    {
        init_compute_memory<float>(&a2, &b2, &c2, &d_a2, &d_b2, &d_c2, num_threads_per_tb);
        auto kernel_args2 = std::make_tuple(d_a2, d_b2, d_c2, num_itrs);

        if (mode == 2)
        {
            cout << "----------------------------------------" << endl;
            cout << "Running two compute kernel sequentially - " << launch_config << endl;
            float avg_seq = run_kernels_sequential(compute_kernel, compute_kernel, gridDim, blockDim, kernel_args1, gridDim, blockDim, kernel_args2, num_runs);
        }
        else
        {
            cout << "----------------------------------------" << endl;
            cout << "Running two compute kernel colocated using CUDA streams - " << launch_config << endl;
            float avg_col = run_kernels_colocated(compute_kernel, compute_kernel, gridDim, blockDim, kernel_args1, gridDim, blockDim, kernel_args2, num_runs);
        }

        free_compute_memory<float>(&a2, &b2, &c2, &d_a2, &d_b2, &d_c2);
    }
    else
    {
        cerr << "Invalid mode!" << endl;
        exit(1);
    }

    free_compute_memory<float>(&a1, &b1, &c1, &d_a1, &d_b1, &d_c1);

    cout << "Done!" << endl;
    return 0;
}