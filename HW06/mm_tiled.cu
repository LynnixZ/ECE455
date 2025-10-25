template <typename T>
__global__ void mm_tiled(const T* A, const T* B, T* C, int N) {
__shared__ T tile_A[TILE_SIZE][TILE_SIZE];
__shared__ T tile_B[TILE_SIZE][TILE_SIZE];
int row = blockIdx.y * TILE_SIZE + threadIdx.y;
int col = blockIdx.x * TILE_SIZE + threadIdx.x;
T val = 0;
for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
if (row < N && (t * TILE_SIZE + threadIdx.x) < N)
tile_A[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
else
tile_A[threadIdx.y][threadIdx.x] = 0;
if (col < N && (t * TILE_SIZE + threadIdx.y) < N)
tile_B[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
else
tile_B[threadIdx.y][threadIdx.x] = 0;
__syncthreads();
for (int k = 0; k < TILE_SIZE; ++k)
val += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
__syncthreads();
}
if (row < N && col < N)
C[row * N + col] = val;
}
