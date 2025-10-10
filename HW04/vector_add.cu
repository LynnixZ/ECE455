#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

int main() {
    const int N = 1'000'000;
    const size_t size = N * sizeof(float);

    // host buffers
    std::vector<float> h_A(N, 1.0f), h_B(N, 2.0f), h_C(N);

    // device buffers
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost);

    printf("C[0]=%f  C[N-1]=%f\n", h_C[0], h_C[N-1]);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
