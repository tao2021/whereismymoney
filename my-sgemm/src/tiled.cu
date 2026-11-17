#include "sgemm.h"

__global__ void tiled_kernel(const float* a, const float* b, float* c, int m, int n, int k) {
    __shared__ float as[16][16], bs[16][16];
    int x = threadIdx.x, y = threadIdx.y;
    int row = blockIdx.y*16+y, col = blockIdx.x*16+x;
    float sum = 0;
    for (int p = 0; p < k; p += 16) {
        as[y][x] = row < m && p+x < k ? a[size_t(row)*k+p+x] : 0;
        bs[y][x] = p+y < k && col < n ? b[size_t(p+y)*n+col] : 0;
        __syncthreads();
        #pragma unroll
        for (int t = 0; t < 16; ++t) sum = fmaf(as[y][t],bs[t][x],sum);
        __syncthreads(); // All readers must finish before the next overwrite.
    }
    if (row < m && col < n) c[size_t(row)*n+col] = sum;
}
void launch_tiled(const float* a, const float* b, float* c, int m, int n, int k, cudaStream_t s) {
    tiled_kernel<<<dim3((n+15)/16, (m+15)/16), dim3(16,16), 0, s>>>(a,b,c,m,n,k);
}
