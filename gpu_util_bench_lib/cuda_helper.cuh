#include <algorithm>
#include <nvml.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <tuple>
#include <stdio.h>

using namespace std;

#define CHECK_NVML_ERROR(result)                                       \
    {                                                                  \
        if (result != NVML_SUCCESS)                                    \
        {                                                              \
            cerr << "NVML Error: " << nvmlErrorString(result) << endl; \
            nvmlShutdown();                                            \
            exit(1);                                                   \
        }                                                              \
    }

#define CUDACHECK(cmd)                                                   \
    do                                                                   \
    {                                                                    \
        cudaError_t e = cmd;                                             \
        if (e != cudaSuccess)                                            \
        {                                                                \
            cout << "Failed: Cuda error " << __FILE__ << ":" << __LINE__ \
                 << cudaGetErrorString(e) << endl;                       \
            throw runtime_error("CUDA FAILURE - THROW ERROR!");          \
        }                                                                \
    } while (0)


bool is_default_stream_complete()
{
    cudaError_t err = cudaStreamQuery(0);
    if (err == cudaSuccess)
    {
        return true;
    }
    else if (err == cudaErrorNotReady)
    {
        return false;
    }
    else
    {
        cerr << "CUDA error: " << cudaGetErrorString(err) << endl;
        nvmlShutdown();
        exit(1);
    }
}


template <typename T>
void init_compute_memory(T **a, T **b, T **c, T **d_a, T **d_b, T **d_c, int num_elems)
{
    size_t size_bytes = num_elems * sizeof(T);

    *a = (T *)malloc(size_bytes);
    *b = (T *)malloc(size_bytes);
    *c = (T *)malloc(size_bytes);

    CUDACHECK(cudaMalloc(d_a, size_bytes));
    CUDACHECK(cudaMalloc(d_b, size_bytes));
    CUDACHECK(cudaMalloc(d_c, size_bytes));

    for (int i = 0; i < num_elems; i++)
    {
        (*a)[i] = 1.0f;
        (*b)[i] = 1.0f;
    }

    CUDACHECK(cudaMemcpy(*d_a, *a, size_bytes, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(*d_b, *b, size_bytes, cudaMemcpyHostToDevice));
}

template <typename T>
void free_compute_memory(T **a, T **b, T **c, T **d_a, T **d_b, T **d_c)
{
    free(*a);
    free(*b);
    free(*c);
    CUDACHECK(cudaFree(*d_a));
    CUDACHECK(cudaFree(*d_b));
    CUDACHECK(cudaFree(*d_c));
}

template <typename T>
void init_copy_memory(T **in, T **out, T **d_in, T **d_out, int num_elems)
{
    size_t size_bytes = num_elems * sizeof(T);

    *in = (T *)malloc(size_bytes);
    *out = (T *)malloc(size_bytes);
    CUDACHECK(cudaMalloc(d_in, size_bytes));
    CUDACHECK(cudaMalloc(d_out, size_bytes));
    for (int i = 0; i < num_elems; i++)
    {
        (*in)[i] = 0;
    }
    CUDACHECK(cudaMemcpy(*d_in, *in, size_bytes, cudaMemcpyHostToDevice));
}

template <typename T>
void free_copy_memory(T **in, T **out, T **d_in, T **d_out)
{
    free(*in);
    free(*out);
    CUDACHECK(cudaFree(*d_in));
    CUDACHECK(cudaFree(*d_out));
}


void profile_with_nvml(nvmlDevice_t device)
{
    nvmlUtilization_t info;
    CHECK_NVML_ERROR(nvmlDeviceGetUtilizationRates(device, &info));
    cout << "SM util: " << info.gpu << ", Mem util: " << info.memory << endl;
}


template <typename K1, typename... Args1>
void run_kernel_alone_with_nvml(K1 kernel1, dim3 gridDim1, dim3 blockDim1, std::tuple<Args1...> args1, int num_runs)
{
    // runs kernel alone and profiles its execution with nvml
    // initialize nvml
    int device_id = 0;
    nvmlDevice_t device;
    CHECK_NVML_ERROR(nvmlInit());
    CHECK_NVML_ERROR(nvmlDeviceGetHandleByIndex(device_id, &device));

    for (int i = 0; i < num_runs; i++)
    {

        std::apply([&](auto &&...args)
                   { kernel1<<<gridDim1, blockDim1>>>(args...); }, args1);

        while (!is_default_stream_complete())
        {
            profile_with_nvml(device);
            usleep(200000);
        }

        CUDACHECK(cudaDeviceSynchronize());
    }

    // shutdown nvml
    CHECK_NVML_ERROR(nvmlShutdown());
}


template <typename K1, typename... Args1>
float run_kernel_alone(K1 kernel1, dim3 gridDim1, dim3 blockDim1, std::tuple<Args1...> args1, int num_runs)
{
    // runs kernel alone and profiles its latency
    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));

    std::vector<float> duration;

    for (int i = 0; i < num_runs; i++)
    {
        CUDACHECK(cudaEventRecord(start));

        std::apply([&](auto &&...args)
                   { kernel1<<<gridDim1, blockDim1>>>(args...); }, args1);

        CUDACHECK(cudaEventRecord(stop));
        CUDACHECK(cudaDeviceSynchronize());

        float lat;
        CUDACHECK(cudaEventElapsedTime(&lat, start, stop));
        duration.push_back(lat);
        std::cout << "Alone time is " << lat << " ms" << std::endl;
    }

    float lat_med;
    std::sort(duration.begin(), duration.end());
    if (num_runs > 1)
        lat_med = (duration[(num_runs / 2) - 1] + duration[num_runs / 2]) / 2;
    else
        lat_med = duration[0];

    std::cout << "Avg alone time is " << lat_med << " ms" << std::endl;

    CUDACHECK(cudaEventDestroy(start));
    CUDACHECK(cudaEventDestroy(stop));

    return lat_med;
}

template <typename K1, typename K2, typename... Args1, typename... Args2>
float run_kernels_sequential(K1 kernel1, K2 kernel2,
                             dim3 gridDim1, dim3 blockDim1, std::tuple<Args1...> args1,
                             dim3 gridDim2, dim3 blockDim2, std::tuple<Args2...> args2,
                             int num_runs)
{
    // runs kernels sequentially and profiles their latency
    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));

    std::vector<float> duration;

    for (int i = 0; i < num_runs; i++)
    {
        CUDACHECK(cudaEventRecord(start));

        std::apply([&](auto &&...args)
                   { kernel1<<<gridDim1, blockDim1>>>(args...); }, args1);
        std::apply([&](auto &&...args)
                   { kernel2<<<gridDim2, blockDim2>>>(args...); }, args2);

        CUDACHECK(cudaEventRecord(stop));
        CUDACHECK(cudaDeviceSynchronize());

        float lat;
        CUDACHECK(cudaEventElapsedTime(&lat, start, stop));
        duration.push_back(lat);
        std::cout << "Sequential time is " << lat << " ms" << std::endl;
    }

    float lat_med;
    std::sort(duration.begin(), duration.end());
    if (num_runs > 1)
        lat_med = (duration[(num_runs / 2) - 1] + duration[num_runs / 2]) / 2;
    else
        lat_med = duration[0];

    std::cout << "Avg sequential time is " << lat_med << " ms" << std::endl;

    CUDACHECK(cudaEventDestroy(start));
    CUDACHECK(cudaEventDestroy(stop));

    return lat_med;
}

template <typename K1, typename K2, typename... Args1, typename... Args2>
float run_kernels_colocated(K1 kernel1, K2 kernel2,
                            dim3 gridDim1, dim3 blockDim1, std::tuple<Args1...> args1,
                            dim3 gridDim2, dim3 blockDim2, std::tuple<Args2...> args2,
                            int num_runs)
{
    // runs kernels colocated using CUDA streams and profiles their latency
    cudaStream_t stream1, stream2;
    CUDACHECK(cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking));
    CUDACHECK(cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking));

    cudaEvent_t start1, stop1, start2, stop2;
    CUDACHECK(cudaEventCreate(&start1));
    CUDACHECK(cudaEventCreate(&stop1));
    CUDACHECK(cudaEventCreate(&start2));
    CUDACHECK(cudaEventCreate(&stop2));

    std::vector<float> duration;

    // run kernels once to load them into memory, avoid lazy loading
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/#lazy-loading
    std::apply([&](auto &&...args)
               { kernel1<<<gridDim1, blockDim1, 0, stream1>>>(args...); }, args1);
    std::apply([&](auto &&...args)
               { kernel2<<<gridDim2, blockDim2, 0, stream2>>>(args...); }, args2);
    CUDACHECK(cudaDeviceSynchronize());

    for (int i = 0; i < num_runs; i++)
    {
        CUDACHECK(cudaEventRecord(start1, stream1));
        std::apply([&](auto &&...args)
                   { kernel1<<<gridDim1, blockDim1, 0, stream1>>>(args...); }, args1);
        CUDACHECK(cudaEventRecord(stop1, stream1));
        CUDACHECK(cudaEventRecord(start2, stream2));
        std::apply([&](auto &&...args)
                   { kernel2<<<gridDim2, blockDim2, 0, stream2>>>(args...); }, args2);
        CUDACHECK(cudaEventRecord(stop2, stream2));

        CUDACHECK(cudaDeviceSynchronize());

        float time1, time2, makespan1, makespan2;
        CUDACHECK(cudaEventElapsedTime(&time1, start1, stop1));
        CUDACHECK(cudaEventElapsedTime(&time2, start2, stop2));
        CUDACHECK(cudaEventElapsedTime(&makespan1, start1, stop2));
        CUDACHECK(cudaEventElapsedTime(&makespan2, start2, stop1));

        makespan1 = max(makespan1, makespan2);
        makespan1 = max(makespan1, time1);
        makespan1 = max(makespan1, time2);
        duration.push_back(makespan1);
        cout << "Colocated time is " << makespan1 << " ms" << endl;
    }

    float lat_med;
    std::sort(duration.begin(), duration.end());
    if (num_runs > 1)
        lat_med = (duration[(num_runs / 2) - 1] + duration[num_runs / 2]) / 2;
    else
        lat_med = duration[0];

    std::cout << "Avg colocated time is " << lat_med << " ms" << std::endl;

    CUDACHECK(cudaEventDestroy(start1));
    CUDACHECK(cudaEventDestroy(stop1));
    CUDACHECK(cudaEventDestroy(start2));
    CUDACHECK(cudaEventDestroy(stop2));
    CUDACHECK(cudaStreamDestroy(stream1));
    CUDACHECK(cudaStreamDestroy(stream2));

    return lat_med;
}