#include "sgemm.h"

__global__ void naive_kernel(const float* a, const float* b, float* c, int m, int n, int k) {
    int row = blockIdx.y * 16 + threadIdx.y;
    int col = blockIdx.x * 16 + threadIdx.x;
    if (row >= m || col >= n) return;
    float sum = 0;
    for (int p = 0; p < k; ++p)
        sum = fmaf(a[size_t(row)*k+p], b[size_t(p)*n+col], sum);
    c[size_t(row)*n+col] = sum;
}
void launch_naive(const float* a, const float* b, float* c, int m, int n, int k, cudaStream_t s) {
    naive_kernel<<<dim3((n+15)/16, (m+15)/16), dim3(16,16), 0, s>>>(a,b,c,m,n,k);
}
