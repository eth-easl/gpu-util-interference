#include <iostream>
#include "kernels.cuh"
#include "cuda_helper.cuh"

using namespace std;

typedef void (*compute_kernel_t)(float *, float *, float *, long long);

compute_kernel_t get_compute_kernel(int ilp)
{
    switch (ilp)
    {
    case 1:
        return mul_fp32_ilp1;
    case 2:
        return mul_fp32_ilp2;
    case 3:
        return mul_fp32_ilp3;
    case 4:
        return mul_fp32_ilp4;
    default:
        cerr << "Invalid ILP value" << endl;
        exit(1);
    }
}

int main(int argc, char **argv)
{
    if (argc < 8)
    {
        cerr << "Please provide 7 arguments" << std::endl;
        return 1;
    }

    int mode = stoi(argv[1]);                    // 0: profile, 1: run alone, 2: run sequential, 3: run colocated
    int kernel_id = stoi(argv[2]);               // 0: copy, 1: ilp1, 2: ilp2, 3: ilp3, 4: ilp4
    int num_threads_per_tb_copy = stoi(argv[3]); // number of threads per block for copy kernel
    int num_threads_per_tb_comp = stoi(argv[4]); // number of threads per block for compute kernel
    long long num_itrs_copy = stol(argv[5]);     // number of iterations for copy kernel
    long long num_itrs_comp = stol(argv[6]);     // number of iterations for compute kernel
    long long num_bytes = stol(argv[7]);         // number of bytes to copy

    long long num_elems_copy = num_bytes / sizeof(float); // number of elements to copy
    int num_elems_comp = num_threads_per_tb_comp;

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
    dim3 blockDimCopy(num_threads_per_tb_copy);
    dim3 blockDimComp(num_threads_per_tb_comp);
    string launch_config_copy = "Launch Config Copy: (" + to_string(num_sms) + ", " + to_string(num_threads_per_tb_copy) + ")";
    string launch_config_comp = "Launch Config Compute: (" + to_string(num_sms) + ", " + to_string(num_threads_per_tb_comp) + ")";


    if (mode == 0 || mode == 1)
    {
        if (kernel_id == 0)
        {
            // run copy kernel alone
            cout << "------------------------------------" << endl;
            cout << "Running copy kernel alone - " << launch_config_copy << endl;

            float *in, *out, *d_in, *d_out;
            init_copy_memory<float>(&in, &out, &d_in, &d_out, num_elems_copy);
            auto copy_args = std::make_tuple(d_in, d_out, num_elems_copy, num_itrs_copy);
            run_kernel_alone(copy_kernel, gridDim, blockDimCopy, copy_args, num_runs);
            free_copy_memory<float>(&in, &out, &d_in, &d_out);
        }
        else
        {
            // run compute kernel alone
            cout << "------------------------------------" << endl;
            cout << "Running compute kernel with ILP " << kernel_id << " alone - " << launch_config_comp << endl;

            auto comp_kernel = get_compute_kernel(kernel_id);
            float *a, *b, *c, *d_a, *d_b, *d_c;
            init_compute_memory<float>(&a, &b, &c, &d_a, &d_b, &d_c, num_elems_comp);
            auto comp_args = std::make_tuple(d_a, d_b, d_c, num_itrs_comp);
            run_kernel_alone(comp_kernel, gridDim, blockDimComp, comp_args, num_runs);
            free_compute_memory<float>(&a, &b, &c, &d_a, &d_b, &d_c);
        }
    }
    else if (mode == 2 || mode == 3)
    {
        float *a, *b, *c, *d_a, *d_b, *d_c;
        float *in, *out, *d_in, *d_out;

        init_compute_memory<float>(&a, &b, &c, &d_a, &d_b, &d_c, num_elems_comp);
        init_copy_memory<float>(&in, &out, &d_in, &d_out, num_elems_copy);

        auto comp_kernel = get_compute_kernel(kernel_id);
        auto comp_args = std::make_tuple(d_a, d_b, d_c, num_itrs_comp);
        auto copy_args = std::make_tuple(d_in, d_out, num_elems_copy, num_itrs_copy);

        if (mode == 2)
        {
            cout << "------------------------------------" << endl;
            cout << "Running copy and compute with ILP " << kernel_id << " sequentially - " << launch_config_copy << " " << launch_config_comp << endl;
            run_kernels_sequential(comp_kernel, copy_kernel, gridDim, blockDimComp, comp_args, gridDim, blockDimCopy, copy_args, num_runs);
        }
        else
        {
            cout << "------------------------------------" << endl;
            cout << "Running copy and compute with ILP " << kernel_id << " colocated - " << launch_config_copy << " " << launch_config_comp << endl;
            run_kernels_colocated(comp_kernel, copy_kernel, gridDim, blockDimComp, comp_args, gridDim, blockDimCopy, copy_args, num_runs);
        }

        free_compute_memory<float>(&a, &b, &c, &d_a, &d_b, &d_c);
        free_copy_memory<float>(&in, &out, &d_in, &d_out);
    }
    else
    {
        cerr << "Invalid mode!" << endl;
        exit(1);
    }

    return 0;
}