#ifndef _TUTORIAL1_QUESTION1_H
#define _TUTORIAL1_QUESTION1_H


__global__ void kernelQuestion1_part1()
{
    int tidx = blockIdx.x*blockDim.x + threadIdx.x;

    printf("Hello from thread: %3.1d Local Data: thread %d from block %d\n",
           tidx,threadIdx.x,blockIdx.x);
}

__global__ void kernelQuestion1_part2()
{
    int nthds = blockDim.x * blockDim.y;
    int tidx = blockIdx.x*nthds + threadIdx.y*blockDim.x + threadIdx.x;

    printf("Hello from thread: %3.1d Local Data: thread (%d,%d) from block %d\n",
           tidx,threadIdx.x,threadIdx.y,blockIdx.x);
}

__global__ void kernelQuestion1_part3()
{
    int bidx = blockIdx.y*gridDim.x + blockIdx.x;
    int nthds = blockDim.x*blockDim.y;
    int tidx = bidx*nthds + threadIdx.y*blockDim.x + threadIdx.x;

    printf("Hello from thread: %3.1d Local Data: thread (%d,%d) from block (%d,%d)\n",
           tidx,threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y);
}

void question1()
{
    cout << endl;
    cout << "Tutorial 1, Question 1" << endl;

    kernelQuestion1_part1<<<3,5>>>();
    cudaDeviceSynchronize();
    cout << endl << endl;

    dim3 B;
    B.x = 2;
    B.y = 5;

    kernelQuestion1_part2<<<3,B>>>();
    cudaDeviceSynchronize();
    cout << endl << endl;

    dim3 G;
    G.x = 2;
    G.y = 3;

    kernelQuestion1_part3<<<G,B>>>();
    cudaDeviceSynchronize();
    cout << endl << endl;
}


#endif
