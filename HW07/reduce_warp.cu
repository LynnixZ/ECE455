__device__ int warp_reduce_sum ( int val ) {
// Each iteration halves the number of active lanes .
// Example : for offset =16 , lanes [0..15] add lanes [16..31] ’ s values .
// The data moves between threads using warp shuffle instructions .
for ( int offset = 1 6 ; offset > 0 ; offset > >= 1 )
val += __shfl_down_sync ( 0 xFFFFFFFF , val , offset ) ;
// After this loop , lane 0 of the warp holds the total sum of that
warp .
return val ;
}
__global__ void reduce_warp ( const int * in , int * out , size_t num_elems ) {
unsigned int idx = blockIdx . x * blockDim . x + threadIdx . x ;
int val = ( idx < num_elems ) ? in [ idx ] : 0 ;
// --- Perform warp - level reduction ---
// Each warp (32 threads ) computes a local partial sum .
val = warp_reduce_sum ( val ) ;
// --- Write partial results ---
// Only the first thread in each warp ( lane 0) performs the global
atomic add .
// This prevents multiple threads in the same warp from writing
duplicates .
if (( threadIdx . x & 3 1 ) == 0 )
atomicAdd ( out , val ) ;
}
