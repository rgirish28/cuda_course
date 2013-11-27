#ifndef _TUTORIAL3_QUESTION1_H
#define _TUTORIAL3_QUESTION1_H



__global__ void kernelQuestion1(
    float S0,
    float r,
    float T,
    float sigma,
    float K,
    int n,
    float *d_Z,
    float *d_out)
{
    int thdsPerBlk = blockDim.x;	// Number of threads per block
    int tidxKernel = blockIdx.x*thdsPerBlk + threadIdx.x;
    const int totalThds = blockDim.x*gridDim.x;

    extern __shared__ float partials[];

    float sum = 0.0f;
    float sqT = sqrt(T);
    for(int zIdx=tidxKernel ; zIdx < n; zIdx += totalThds) {
        float ST = S0*exp( (r-0.5f*sigma*sigma)*T + sigma*sqT*d_Z[zIdx] );
        float x = (ST>K ? ST-K : 0.0f);
        x = exp(-r*T)*x;
        sum += x;
    }

    partials[threadIdx.x] = sum;
    __syncthreads();

    if(threadIdx.x == 0) {
        sum = 0.0f;
        for(int i=0; i<thdsPerBlk; i++) {
            sum += partials[i];
        }
        d_out[blockIdx.x] = sum;
    }
}



void callSinglePrecisionKernel(int n)
{
    int nthds = 672;
    int nblks = 140;

    float
    r = 0.02f,
    T = 1.0f,
    sigma = 0.09f,
    K = 100.0f,
    S0 = 100.0f;

    float *d_Z = NULL;
    float *d_out = NULL;

    check( cudaMalloc((void**)&d_Z, sizeof(float)*n) );
    check( cudaMalloc((void**)&d_out, sizeof(float)*nblks) );

    generateRandomNumbers(n, d_Z);


    cudaEvent_t start, stop;
    check( cudaEventCreate(&start) );
    check( cudaEventCreate(&stop) );

    check( cudaEventRecord(start) );
    kernelQuestion1<<<nblks, nthds, sizeof(float)*nthds>>>(S0, r, T, sigma, K, n, d_Z, d_out);
    check( cudaEventRecord(stop) );

    check( cudaGetLastError() );

    float *out = new float[nblks];
    check( cudaMemcpy(out, d_out, sizeof(float)*nblks, cudaMemcpyDeviceToHost) );

    float time;
    cudaEventElapsedTime(&time, start, stop);

    float sum = 0.0f;
    for(int i=0; i<nblks; i++) {
        sum += out[i];
    }


    cout << "\taverage=" << sum/n << endl;
    cout << "\tkernel runtime=" << time << "ms" << endl;

    check( cudaFree(d_Z) );
    check( cudaFree(d_out) );
    delete[] out;
    check( cudaEventDestroy(start) );
    check( cudaEventDestroy(stop) );

}



void question1()
{
    cout << endl;
    cout << "Tutorial 2, Question 1" << endl;

    int n = 100000000;
    cout << "Calling Double Precision Kernel: " << endl;
    callDoublePrecisionKernel(n);
    cout << "Calling Single Precision Kernel: " << endl;
    callSinglePrecisionKernel(n);
}


#endif
