#ifndef _TUTORIAL3_QUESTION3_H
#define _TUTORIAL3_QUESTION3_H



__device__ float d_blkSum = 0.0f;

__global__ void kernelQuestion3(
    float S0,
    float r,
    float T,
    float sigma,
    float K,
    int n,
    float *d_Z)
{
    int thdsPerBlk = blockDim.x;	// Number of threads per block
    int tidxKernel = blockIdx.x*thdsPerBlk + threadIdx.x;
    int totalThds = thdsPerBlk*gridDim.x;

    extern __shared__ float partials[];

    float sum = 0.0f;
    float sqT = sqrt(T);
    for(int zIdx=tidxKernel; zIdx<n; zIdx+=totalThds) {
        float ST = S0*exp( (r-0.5f*sigma*sigma)*T + sigma*sqT*d_Z[zIdx] );
        float x = (ST>K ? ST-K : 0.0f);
        x = exp(-r*T)*x;
        sum += x;
    }

    partials[threadIdx.x] = sum;
    __syncthreads();

    if(threadIdx.x == 0) {
        sum = 0.0;
        for(int i=0; i<thdsPerBlk; i++) {
            sum += partials[i];
        }
        atomicAdd(&d_blkSum, sum);
    }
}

void question3()
{
    cout << endl;
    cout << "Tutorial 2, Question 3" << endl;

    int n = 100000000;
    int nthds = 192;
    int nblks = 224;

    float
    r = 0.02f,
    T = 1.0f,
    sigma = 0.09f,
    K = 100.0f,
    S0 = 100.0f;

    float *d_Z = NULL;
    float *d_out = NULL;

    check( cudaMalloc((void**)&d_Z, sizeof(float)*n) );

    generateRandomNumbers(n, d_Z);


    cudaEvent_t start, stop;
    check( cudaEventCreate(&start) );
    check( cudaEventCreate(&stop) );

    check( cudaEventRecord(start) );
    kernelQuestion3<<<nblks, nthds, sizeof(float)*nthds>>>(S0, r, T, sigma, K, n, d_Z);
    check( cudaEventRecord(stop) );

    check( cudaDeviceSynchronize() );
    check( cudaGetLastError() );

    float time;
    cudaEventElapsedTime(&time, start, stop);

    float sum = 0.0f;
    check( cudaMemcpyFromSymbol(&sum, d_blkSum, sizeof(float)) );

    cout << "average=" << sum/n << endl;
    cout << "kernel runtime=" << time << "ms" << endl;

    check( cudaFree(d_Z) );
    check( cudaFree(d_out) );
    check( cudaEventDestroy(start) );
    check( cudaEventDestroy(stop) );
}


#endif
