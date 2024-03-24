#include <stdio.h>
#include <stdlib.h>


int main()
{
    int numElements = 16;
    int totalBytes = numElements * sizeof( int );

    int* deviceArray = 0;
    int* hostArray = 0;

    // malloc host array
    hostArray = ( int* )malloc( totalBytes );

    // cudaMalloc device array
    cudaMalloc( ( void** )&deviceArray, totalBytes );

    // set zeros for cuda array
    cudaMemset( deviceArray, 0, totalBytes );

    // copy content of the device array to the host
    cudaMemcpy( hostArray, deviceArray, totalBytes, cudaMemcpyDeviceToHost );


    // print elements in host array
    for ( int i = 0; i < numElements; ++i )
    {
        printf( "%d: %d\n", i, hostArray[ i ] );
    }

    // free host memory
    free( hostArray );

    // free device memory
    cudaFree( deviceArray );

    return 0;
}