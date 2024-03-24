#include <stdlib.h>
#include <stdio.h>

/* __global__ functions expose a particular form of parallel computation called data parallelism
    The basic idea of data parallelism is to distribute a large task composed of many similar but 
    independent pieces across a set of computational resources. In Cuda, the task to be performed
    is descrided by a __global__ function, and the computational resources are CUDA threads.
*/

// Mapping subtasks to particular device threads
__global__ void kernal( int* i_device_array )
{
    // blockIdx -> index of the block within the grid
    // blockDim -> number of threads in each thread block
    // threadIdx -> index of each cuda thread within its thread block
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    i_device_array[index] = 7;
}

int main()
{
    int num_elements = 256;
    int num_bytes = sizeof(int) * num_elements;

    int* device_array = 0;
    int* host_array = 0;

    // allocate host array
    host_array = ( int* )malloc( num_bytes );

    // cudaMalloc allocate device array
    // cudaMalloc ( void** devPtr, size_t size )
    cudaMalloc( (void**)&device_array, num_bytes );

    int block_size = 128;
    int grid_size = num_elements / block_size;

    kernal<<<grid_size, block_size>>>( device_array );

    // download from device to host
    // cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind )
    cudaMemcpy( host_array, device_array, num_bytes, cudaMemcpyDeviceToHost );

    // print out result 
    for ( int i = 0; i < num_elements; ++i )
    {
        printf( "Element %d is %d.\n", i, host_array[i] );
    }

    free(host_array);
    cudaFree(device_array);
}