#include <cstdio>
#include <cuda_runtime.h>

__global__ void hello_kernel() {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello from thread %d\n", tid);
}

int main() {
    // launch 2 blocks, 4 threads per block -> 8 lines of output
    hello_kernel<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}
