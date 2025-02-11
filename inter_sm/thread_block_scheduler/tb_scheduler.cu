#include <iostream>
#include "kernels.cuh"
#include "cuda_helper.cuh"

using namespace std;

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        cerr << "Please provide 3 arguments!" << endl;
        exit(1);
    }

    int mode = stoi(argv[1]);               // 0: profile, 1: run alone, 2: run sequential, 3: run colocated
    int num_tb = stoi(argv[2]);             // number of thread blocks blocks
    int num_threads_per_tb = stoi(argv[3]); // number of threads per block;
    long long num_itrs = stoll(argv[4]);    // number of iterations

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

    auto kernel = sleep_kernel;
    auto args = std::make_tuple(num_itrs);

    if (mode == 0 || mode == 1)
    {
        cout << "----------------------------------------" << endl;
        cout << "Running sleep kernel alone - " << launch_config << endl;
        run_kernel_alone(kernel, gridDim, blockDim, args, num_runs);
    }
    else if (mode == 2)
    {
        cout << "----------------------------------------" << endl;
        cout << "Running two sleep kernels sequentially - " << launch_config << endl;
        run_kernels_sequential(kernel, kernel, gridDim, blockDim, args, gridDim, blockDim, args, num_runs);
    }
    else if (mode == 3)
    {
        cout << "----------------------------------------" << endl;
        cout << "Running two sleep kernels colocated using CUDA streams - " << launch_config << endl;
        run_kernels_colocated(kernel, kernel, gridDim, blockDim, args, gridDim, blockDim, args, num_runs);
    }
    else
    {
        cerr << "Invalid mode!" << endl;
        exit(1);
    }
}