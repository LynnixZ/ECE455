#include <cstdio>
#include <cuda_runtime.h>

__global__ void print_indices() {
    printf("Block (%d,%d)  Thread (%d,%d)\n",
           blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
}

int main() {
    dim3 blocks(2, 2);   // 2x2 grid
    dim3 threads(2, 3);  // 2x3 block
    print_indices<<<blocks, threads>>>();
    cudaDeviceSynchronize();
    return 0;
}
