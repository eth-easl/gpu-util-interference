#include "kernels.cuh"

__global__ void sleep_kernel(long long num_itr)
{
    for (long long i = 0; i < num_itr; ++i)
    {
        asm volatile("nanosleep.u32 1000;");
    }
}

__global__ void mul_fp32_ilp1(float *a, float *b, float *c, long long num_itr)
{
    float op1 = a[threadIdx.x];
    float op3 = 1.0f;
    for (long long i = 0; i < num_itr; i++)
    {
        op3 = __fmul_rn(op1, op3);
    }
    c[threadIdx.x] = op3;
}

__global__ void mul_fp32_ilp2(float *a, float *b, float *c, long long num_itr)
{
    float op1 = a[threadIdx.x];
    float op2 = b[threadIdx.x];
    float op3 = 1.0f;
    float op4 = 1.0f;
    for (long long i = 0; i < num_itr; i++)
    {
        op3 = __fmul_rn(op1, op3);
        op4 = __fmul_rn(op2, op4);
    }
    c[threadIdx.x] = op3 + op4;
}

__global__ void mul_fp32_ilp3(float *a, float *b, float *c, long long num_itr)
{
    float op1 = a[threadIdx.x];
    float op2 = b[threadIdx.x];
    float op3 = 1.0f;
    float op4 = 1.0f;
    float op5 = 1.0f;
    for (long long i = 0; i < num_itr; i++)
    {
        op3 = __fmul_rn(op1, op3);
        op4 = __fmul_rn(op2, op4);
        op5 = __fmul_rn(op1, op5);
    }
    c[threadIdx.x] = op3 + op4 + op5;
}

__global__ void mul_fp32_ilp4(float *a, float *b, float *c, long long num_itr)
{
    float op1 = a[threadIdx.x];
    float op2 = b[threadIdx.x];
    float op3 = 1.0f;
    float op4 = 1.0f;
    float op5 = 1.0f;
    float op6 = 1.0f;
    for (long long i = 0; i < num_itr; i++)
    {
        op3 = __fmul_rn(op1, op3);
        op4 = __fmul_rn(op2, op4);
        op5 = __fmul_rn(op1, op5);
        op6 = __fmul_rn(op2, op6);
    }
    c[threadIdx.x] = op3 + op4 + op5 + op6;
}

__global__ void mul_fp64_ilp1(double *a, double *b, double *c, long long num_itr)
{
    double op1 = a[threadIdx.x];
    double op3 = 1.0f;
    for (long long i = 0; i < num_itr; i++)
    {
        op3 = __dmul_rn(op1, op3);
    }
    c[threadIdx.x] = op3;
}

__global__ void mul_fp64_ilp2(double *a, double *b, double *c, long long num_itr)
{
    double op1 = a[threadIdx.x];
    double op2 = b[threadIdx.x];
    double op3 = 1.0f;
    double op4 = 1.0f;
    for (long long i = 0; i < num_itr; i++)
    {
        op3 = __dmul_rn(op1, op3);
        op4 = __dmul_rn(op2, op4);
    }
    c[threadIdx.x] = op3 + op4;
}

__global__ void mul_fp64_ilp3(double *a, double *b, double *c, long long num_itr)
{
    double op1 = a[threadIdx.x];
    double op2 = b[threadIdx.x];
    double op3 = 1.0f;
    double op4 = 1.0f;
    double op5 = 1.0f;
    for (long long i = 0; i < num_itr; i++)
    {
        op3 = __dmul_rn(op1, op3);
        op4 = __dmul_rn(op2, op4);
        op5 = __dmul_rn(op1, op5);
    }
    c[threadIdx.x] = op3 + op4 + op5;
}

__global__ void mul_fp64_ilp4(double *a, double *b, double *c, long long num_itr)
{
    double op1 = a[threadIdx.x];
    double op2 = b[threadIdx.x];
    double op3 = 1.0f;
    double op4 = 1.0f;
    double op5 = 1.0f;
    double op6 = 1.0f;
    for (long long i = 0; i < num_itr; i++)
    {
        op3 = __dmul_rn(op1, op3);
        op4 = __dmul_rn(op2, op4);
        op5 = __dmul_rn(op1, op5);
        op6 = __dmul_rn(op2, op6);
    }
    c[threadIdx.x] = op3 + op4 + op5 + op6;
}

__global__ void copy_kernel_per_tb(float *in, float *out, long long num_floats_per_tb, long long num_itrs, int region_size_bytes)
{
    // each thread block copies num_floats_per_tb floats from in to out
    // each thread block operates on separate non-overlapping regions
    // if num_floats_per_tb > region_size_bytes, assign multiple regions to each thread block

    // calculate the number of floats per region
    int floats_per_region = region_size_bytes / sizeof(float);

    // calculate the number of regions per thread block to ensure no overlapping
    int regions_per_tb = (num_floats_per_tb + floats_per_region - 1) / floats_per_region;

    int block_begin = blockIdx.x * regions_per_tb * floats_per_region;
    int block_end = block_begin + num_floats_per_tb;

    for (int i = 0; i < num_itrs; i++)
    {
        for (int j = block_begin + threadIdx.x; j < block_end; j = j + blockDim.x)
        {
            out[j] = in[j];
        }
    }
}

__global__ void copy_kernel(float *in, float *out, long long num_floats, long long num_itr)
{
    // copies in to out, using memory coalescing
    size_t start = threadIdx.x + blockDim.x * blockIdx.x;
    size_t step = gridDim.x * blockDim.x;
    for (size_t j = 0; j < num_itr; j++)
    {
        for (size_t i = start; i < num_floats; i += step)
        {
            out[i] = in[i];
        }
    }
}

__global__ void pmevent_kernel(long long num_itrs){
    // kernel that increments a PM event counter, high IPC with low pipeline utilization
    for (long long i = 0; i < num_itrs; ++i){
        asm volatile("pmevent 0;");
    }
}