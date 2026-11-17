#pragma once
#include <cuda_runtime.h>

// Row-major: A[M,K], B[K,N], C[M,N]. C = A*B; no alpha/beta.
// Positive dimensions; allocations and synchronization belong to the caller.
using Launcher = void (*)(const float*, const float*, float*, int, int, int, cudaStream_t);
void launch_naive(const float*, const float*, float*, int m, int n, int k, cudaStream_t);
void launch_tiled(const float*, const float*, float*, int m, int n, int k, cudaStream_t);
// FP32 inputs rounded to FP16 inside shared-memory staging, FP32 accumulation.
void launch_tensor_core(const float*, const float*, float*, int m, int n, int k, cudaStream_t);
