#include <iostream>
#include "kernels.cuh"
#include "cuda_helper.cuh"

typedef void (*compute_kernel_t)(double *, double *, double *, long long);

compute_kernel_t get_compute_kernel(int ilp)
{
    switch (ilp)
    {
    case 1:
        return mul_fp64_ilp1;
    case 2:
        return mul_fp64_ilp2;
    case 3:
        return mul_fp64_ilp3;
    case 4:
        return mul_fp64_ilp4;
    default:
        cerr << "Invalid ILP value" << endl;
        exit(1);
    }
}

int main(int argc, char **argv)
{
    if (argc < 5)
    {
        cerr << "Please provide 4 arguments" << std::endl;
        return 1;
    }

    int mode = stoi(argv[1]);               // 0: profile, 1: run alone, 2: run sequential, 3: run colocated
    int ilp_level = stoi(argv[2]);          // 1: ilp1, 2: ilp2, 3: ilp3, 4: ilp4
    int num_threads_per_tb = stoi(argv[3]); // number of threads per block for compute kernel
    long long num_itrs = stoll(argv[4]);    // number of iterations for compute kernel

    int num_runs = (mode == 0) ? 1 : 10; // in profiling mode, perform only single run

    int device_id = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    int num_sms = prop.multiProcessorCount;
    if (!prop.concurrentKernels)
    {
        cerr << "Concurrent Kernels not supported on this device." << endl;
        return 1;
    }

    dim3 gridDim(num_sms);
    dim3 blockDim(num_threads_per_tb);
    string launch_config = "Launch Config: (" + to_string(num_sms) + ", " + to_string(num_threads_per_tb) + ")";

    // compute kernel used for experiment
    auto comp_kernel = get_compute_kernel(ilp_level);

    double *a1, *b1, *c1, *d_a1, *d_b1, *d_c1;
    double *a2, *b2, *c2, *d_a2, *d_b2, *d_c2;

    init_compute_memory<double>(&a1, &b1, &c1, &d_a1, &d_b1, &d_c1, num_threads_per_tb);
    auto kernel_args1 = std::make_tuple(d_a1, d_b1, d_c1, num_itrs);

    if (mode == 0 || mode == 1)
    {
        cout << "------------------------------------" << endl;
        cout << "Running compute kernel with ILP " << ilp_level << " alone - " << launch_config << endl;
        run_kernel_alone(comp_kernel, gridDim, blockDim, kernel_args1, num_runs);
    }
    else if (mode == 2 || mode == 3)
    {
        init_compute_memory<double>(&a2, &b2, &c2, &d_a2, &d_b2, &d_c2, num_threads_per_tb);
        auto kernel_args2 = std::make_tuple(d_a2, d_b2, d_c2, num_itrs);

        if (mode == 2)
        {
            cout << "------------------------------------" << endl;
            cout << "Running two compute kernel with ILP " << ilp_level << " sequentially - " << launch_config << endl;
            run_kernels_sequential(comp_kernel, comp_kernel, gridDim, blockDim, kernel_args1, gridDim, blockDim, kernel_args2, num_runs);
        }
        else
        {
            cout << "------------------------------------" << endl;
            cout << "Running two compute kernel with ILP " << ilp_level << " colocated - " << launch_config << endl;
            run_kernels_colocated(comp_kernel, comp_kernel, gridDim, blockDim, kernel_args1, gridDim, blockDim, kernel_args2, num_runs);
        }
    }
    else
    {
        cerr << "Invalid mode!" << endl;
        exit(1);
    }

    free_compute_memory<double>(&a1, &b1, &c1, &d_a1, &d_b1, &d_c1);

    return 0;
}