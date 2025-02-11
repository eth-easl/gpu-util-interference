#pragma once

__global__ void sleep_kernel(long long num_itr);
__global__ void mul_fp32_ilp1(float *a, float *b, float *c, long long num_itr);
__global__ void mul_fp32_ilp2(float *a, float *b, float *c, long long num_itr);
__global__ void mul_fp32_ilp3(float *a, float *b, float *c, long long num_itr);
__global__ void mul_fp32_ilp4(float *a, float *b, float *c, long long num_itr);
__global__ void mul_fp64_ilp1(double *a, double *b, double *c, long long num_itr);
__global__ void mul_fp64_ilp2(double *a, double *b, double *c, long long num_itr);
__global__ void mul_fp64_ilp3(double *a, double *b, double *c, long long num_itr);
__global__ void mul_fp64_ilp4(double *a, double *b, double *c, long long num_itr);
__global__ void copy_kernel_per_tb(float* in, float* out, long long num_floats_per_tb, long long num_itrs, int alignment);
__global__ void copy_kernel(float *in, float *out, long long num_floats, long long num_itr);
__global__ void pmevent_kernel(long long num_itr);