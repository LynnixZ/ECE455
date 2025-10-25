template <typename T>
__global__ void square_shared_kernel(const T* in, T* out, size_t N) {
__shared__ T tile[BLOCK_DIM];
size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx >= N) return;
tile[threadIdx.x] = in[idx];
__syncthreads();
tile[threadIdx.x] = tile[threadIdx.x] * tile[threadIdx.x];
__syncthreads();
out[idx] = tile[threadIdx.x];
}
