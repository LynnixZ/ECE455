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

    std::vector<float> A(N, 1.0f), B(N, 2.0f), C(N, 0.0f);
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // two streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    const int half = N / 2;
    const size_t half_size = size / 2;

    // async H2D
    cudaMemcpyAsync(d_A,        A.data(),        half_size, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_B,        B.data(),        half_size, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_A + half, A.data() + half, half_size, cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(d_B + half, B.data() + half, half_size, cudaMemcpyHostToDevice, stream2);

    int threads = 256;
    int blocks_half = (half + threads - 1) / threads;

    // launch on streams (each stream works on one half)
    vector_add<<<blocks_half, threads, 0, stream1>>>(d_A,        d_B,        d_C,        half);
    vector_add<<<blocks_half, threads, 0, stream2>>>(d_A + half, d_B + half, d_C + half, half);

    // async D2H
    cudaMemcpyAsync(C.data(),        d_C,        half_size, cudaMemcpyDeviceToHost, stream1);
    cudaMemcpyAsync(C.data() + half, d_C + half, half_size, cudaMemcpyDeviceToHost, stream2);

    // sync
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    printf("C[0]=%f  C[N-1]=%f\n", C[0], C[N-1]);

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
