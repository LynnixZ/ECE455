__global__ void reduce_naive ( const int * in , int * out , size_t n_elems ) {
unsigned int idx = blockIdx . x * blockDim . x + threadIdx . x ;
if ( idx < n_elems )
atomicAdd ( out , in [ idx ]) ;
}
