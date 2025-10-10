#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

__global__ void saxpy(int n, float a, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

int main() {
    const int N = 1'000'000;
    const size_t size = N * sizeof(float);

    std::vector<float> x(N, 1.0f), y(N, 2.0f);
    float *d_x = nullptr, *d_y = nullptr;
    cudaMalloc(&d_x, size);
    cudaMalloc(&d_y, size);

    cudaMemcpy(d_x, x.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y.data(), size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    saxpy<<<blocksPerGrid, threadsPerBlock>>>(N, 2.0f, d_x, d_y);

    cudaMemcpy(y.data(), d_y, size, cudaMemcpyDeviceToHost);
    printf("y[0]=%f  y[N-1]=%f\n", y[0], y[N-1]);

    cudaFree(d_x); cudaFree(d_y);
    return 0;
}
