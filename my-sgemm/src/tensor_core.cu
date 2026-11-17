#include "sgemm.h"
#include <cuda_fp16.h>
#include <mma.h>

// One warp per output tile: intentionally small and readable, not peak throughput.
__global__ void tensor_core_kernel(const float* a, const float* b, float* c, int m, int n, int k) {
    using namespace nvcuda;
    __shared__ __align__(32) half as[256];
    __shared__ __align__(32) half bs[256];
    __shared__ __align__(32) float cs[256];
    int lane = threadIdx.x, row = blockIdx.y*16, col = blockIdx.x*16;
    wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> af;
    wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> bf;
    wmma::fragment<wmma::accumulator,16,16,16,float> cf;
    wmma::fill_fragment(cf, 0.0f);
    for (int p = 0; p < k; p += 16) {
        for (int i = lane; i < 256; i += 32) {
            int r = i/16, t = i%16;
            as[i] = __float2half_rn(row+r < m && p+t < k ? a[size_t(row+r)*k+p+t] : 0);
            bs[i] = __float2half_rn(p+r < k && col+t < n ? b[size_t(p+r)*n+col+t] : 0);
        }
        __syncwarp();
        wmma::load_matrix_sync(af, as, 16);
        wmma::load_matrix_sync(bf, bs, 16);
        wmma::mma_sync(cf, af, bf, cf);
        __syncwarp();
    }
    wmma::store_matrix_sync(cs, cf, 16, wmma::mem_row_major);
    __syncwarp();
    for (int i = lane; i < 256; i += 32)
        if (row+i/16 < m && col+i%16 < n) c[size_t(row+i/16)*n+col+i%16] = cs[i];
}
void launch_tensor_core(const float* a, const float* b, float* c, int m, int n, int k, cudaStream_t s) {
    tensor_core_kernel<<<dim3((n+15)/16, (m+15)/16),32,0,s>>>(a,b,c,m,n,k);
}
