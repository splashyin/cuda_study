#include <stdio.h>

// cuda headers
#include "../utils/cuPrintf.cu"

__global__ void device_greetings()
{
    cuPrintf( "Hello, world from the device.\n" );
}

__global__ void device_greetings_parallel()
{
    cuPrintf( "\tHello, world from the device.\n" );
}

int main( void )
{
    printf( "Hello, world from the host.\n" );

    // Initialize cuPrintf
    cudaPrintfInit();

    // Launch kernal with 10 Blocks, and 64 threads per block (640 threads in total)
    device_greetings_parallel<<<10, 64>>>();

    // Launch kernal with a single thread (Block 1, single thread)
    device_greetings<<<1, 1>>>();

    // Display the device's printing
    cudaPrintfDisplay();

    // Clean up after print
    cudaPrintfEnd();

    return 0;
}