#include <iostream>
#include "kernels.cuh"
#include "cuda_helper.cuh"

using namespace std;

int main(int argc, char *argv[])
{
    if (argc < 5)
    {
        cerr << "Please provide 4 arguments!" << endl;
        exit(1);
    }

    int mode = stoi(argv[1]);               // 0: profile, 1: run alone, 2: run sequential, 3: run colocated
    int num_threads_per_tb = stoi(argv[2]); // number of threads per block for each kernel launch
    int num_iters_comp = stoi(argv[3]);     // number of iterations for compute kernel
    int num_iters_copy = stoi(argv[4]);     // number of iterations for memory kernel

    long long num_comp_elems = num_threads_per_tb;
    long long num_copy_elems = 1024 * 1024 * 1024; // 4 GB

    int num_runs = (mode == 0) ? 1 : 10; // in profiling mode, perform only single run

    // read number of visible SMs from device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int num_sms = prop.multiProcessorCount;

    if (!prop.concurrentKernels)
    {
        cerr << "Concurrent Kernels not supported on this device." << endl;
        return 1;
    }

    // run both kernels with one thread block on each SM
    dim3 gridDim(num_sms);
    dim3 blockDim(num_threads_per_tb);
    string launch_config = "Launch Config: (" + to_string(num_sms) + ", " + to_string(num_threads_per_tb) + ")";

    float *a, *b, *c, *d_a, *d_b, *d_c;
    float *in, *out, *d_in, *d_out;

    init_compute_memory<float>(&a, &b, &c, &d_a, &d_b, &d_c, num_comp_elems);
    init_copy_memory<float>(&in, &out, &d_in, &d_out, num_copy_elems);

    // compute and copy kernel used for experiment
    auto compute_kernel = mul_fp32_ilp4;
    auto memory_kernel = copy_kernel;

    auto args_compute = std::make_tuple(d_a, d_b, d_c, num_iters_comp);
    auto args_memory = std::make_tuple(d_in, d_out, num_copy_elems, num_iters_copy);

    if (mode == 0 || mode == 1)
    {
        cout << "----------------------------------------" << endl;
        cout << "Running compute kernel alone - " << launch_config << endl;
        run_kernel_alone(compute_kernel, gridDim, blockDim, args_compute, num_runs);

        cout << "----------------------------------------" << endl;
        cout << "Running copy kernel alone - " << launch_config << endl;
        run_kernel_alone(memory_kernel, gridDim, blockDim, args_memory, num_runs);
    }
    else if (mode == 2)
    {
        cout << "----------------------------------------" << endl;
        cout << "Running compute and copy kernel sequentially - " << launch_config << endl;
        run_kernels_sequential(compute_kernel, memory_kernel, gridDim, blockDim, args_compute, gridDim, blockDim, args_memory, num_runs);
    }
    else if (mode == 3)
    {
        cout << "----------------------------------------" << endl;
        cout << "Running compute and copy kernel colocated - " << launch_config << endl;
        run_kernels_colocated(compute_kernel, memory_kernel, gridDim, blockDim, args_compute, gridDim, blockDim, args_memory, num_runs);
    }
    else
    {
        cerr << "Invalid mode!" << endl;
        exit(1);
    }

    free_compute_memory<float>(&a, &b, &c, &d_a, &d_b, &d_c);
    free_copy_memory<float>(&in, &out, &d_in, &d_out);
}