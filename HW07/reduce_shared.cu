__global__ void reduce_shared ( const int * in , int * out , size_t n_elems ) {
// Each block stores partial results in on - chip shared memory
__shared__ int sdata [ BLOCK_DIM ];
unsigned int tid = threadIdx . x ;
unsigned int idx = blockIdx . x * blockDim . x + threadIdx . x ;
// Load one element per thread from global to shared memory
// Threads beyond the valid range write 0 to avoid out - of - bound reads
int x = ( idx < n_elems ) ? in [ idx ] : 0 ;
sdata [ tid ] = x ;
__syncthreads () ; // Wait for all threads to finish loading
// --- In - block tree reduction ---
// Start with stride = half block size , then repeatedly halve .
// Each active thread adds the element stride positions ahead .
// Example : stride =128 == > thread 0 adds thread 128 ’ s value , etc .
for ( unsigned int stride = blockDim . x / 2 ; stride > 0 ; stride > >= 1 ) {
if ( tid < stride )
sdata [ tid ] += sdata [ tid + stride ];
// Synchronize to make sure all additions complete
// before using the updated shared memory in the next step .
__syncthreads () ;
}
// After the loop , thread 0 holds the sum of all elements in this
block .
// It atomically adds that partial result to the global output .
if ( tid == 0 )
atomicAdd ( out , sdata [ 0 ]) ;
}
