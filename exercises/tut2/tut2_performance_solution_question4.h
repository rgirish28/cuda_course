#ifndef _TUTORIAL3_QUESTION4_H
#define _TUTORIAL3_QUESTION4_H



__device__ float d_thdSum = 0.0f;

__global__ void kernel1Question4(
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

    float sum = 0.0f;
    float sqT = sqrt(T);
    for(int zIdx=tidxKernel; zIdx<n; zIdx+=totalThds) {
        float ST = S0*exp( (r-0.5f*sigma*sigma)*T + sigma*sqT*d_Z[zIdx] );
        float x = (ST>K ? ST-K : 0.0f);
        x = exp(-r*T)*x;
        sum += x;
    }

    atomicAdd(&d_thdSum, sum);
}


__global__ void kernel2Question4(
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

    float sqT = sqrt(T);
    for(int zIdx=tidxKernel; zIdx<n; zIdx+=totalThds) {
        float ST = S0*exp( (r-0.5f*sigma*sigma)*T + sigma*sqT*d_Z[zIdx] );
        float x = (ST>K ? ST-K : 0.0f);
        x = exp(-r*T)*x;
        atomicAdd(&d_thdSum, x);
    }

}


void callKernel1(int n)
{
    int nthds = 192;
    int nblks = 224;

    float
    r = 0.02f,
    T = 1.0f,
    sigma = 0.09f,
    K = 100.0f,
    S0 = 100.0f;

    float *d_Z = NULL;

    check( cudaMalloc((void**)&d_Z, sizeof(float)*n) );

    generateRandomNumbers(n, d_Z);

    cudaEvent_t start, stop;
    check( cudaEventCreate(&start) );
    check( cudaEventCreate(&stop) );

    check( cudaEventRecord(start) );
    kernel1Question4<<<nblks, nthds>>>(S0, r, T, sigma, K, n, d_Z);
    check( cudaEventRecord(stop) );

    check( cudaGetLastError() );

    cudaDeviceSynchronize();
    float time;
    cudaEventElapsedTime(&time, start, stop);

    float sum;
    check( cudaMemcpyFromSymbol(&sum, d_thdSum, sizeof(float)) );


    cout << "\taverage=" << sum/n << endl;
    cout << "\tkernel runtime=" << time << "ms" << endl;

    check( cudaFree(d_Z) );
    check( cudaEventDestroy(start) );
    check( cudaEventDestroy(stop) );

}
void callKernel2(int n)
{
    int nthds = 192;
    int nblks = 224;

    float
    r = 0.02f,
    T = 1.0f,
    sigma = 0.09f,
    K = 100.0f,
    S0 = 100.0f;

    float *d_Z = NULL;

    check( cudaMalloc((void**)&d_Z, sizeof(float)*n) );

    generateRandomNumbers(n, d_Z);

    cudaEvent_t start, stop;
    check( cudaEventCreate(&start) );
    check( cudaEventCreate(&stop) );

    check( cudaEventRecord(start) );
    kernel2Question4<<<nblks, nthds>>>(S0, r, T, sigma, K, n, d_Z);
    check( cudaEventRecord(stop) );

    check( cudaGetLastError() );

    cudaDeviceSynchronize();
    float time;
    cudaEventElapsedTime(&time, start, stop);

    float sum;
    check( cudaMemcpyFromSymbol(&sum, d_thdSum, sizeof(float)) );


    cout << "\taverage=" << sum/n << endl;
    cout << "\tkernel runtime=" << time << "ms" << endl;

    check( cudaFree(d_Z) );
    check( cudaEventDestroy(start) );
    check( cudaEventDestroy(stop) );
}

void question4()
{
    cout << endl;
    cout << "Tutorial 2, Question 4" << endl;
    int n = 100000000;

    cout << "Kernel with atomics at end of compute loop:" << endl;
    callKernel1(n);

    float zero = 0.0f;
    check( cudaMemcpyToSymbol(d_thdSum, &zero, sizeof(float)) );

    cout << "Kernel with atomics inside compute loop:" << endl;
    callKernel2(n);
}


#endif
