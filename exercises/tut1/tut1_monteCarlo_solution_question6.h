#ifndef _TUTORIAL1_QUESTION6_H
#define _TUTORIAL1_QUESTION6_H


__global__ void kernelQuestion6(
    double S0,
    double r,
    double T,
    double sigma,
    double K,
    int n,
    double *d_Z,
    double *d_out)
{
    int thdsPerBlk = blockDim.x;
    int tidxKernel = blockIdx.x*thdsPerBlk + threadIdx.x;
    int totalThds = gridDim.x*blockDim.x;
    extern __shared__ double partials[];

    double sum = 0.0;
    double sqT = sqrt(T);
    for(int zIdx=tidxKernel; zIdx<n; zIdx+=totalThds) {
        double ST = S0*exp( (r-0.5*sigma*sigma)*T + sigma*sqT*d_Z[zIdx] );
        double x = (ST>K ? ST-K : 0.0);
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
        d_out[blockIdx.x] = sum;
    }
}

void question6()
{
    cout << endl;
    cout << "Tutorial 1, Question 6" << endl;

    int n = 100000000;
    int nthds = 672;
    int nblks = 140;

    double
    r = 0.02,
    T = 1.0,
    sigma = 0.09,
    K = 100.0,
    S0 = 100.0;

    double *d_Z = NULL;
    double *d_out = NULL;

    check( cudaMalloc((void**)&d_Z, sizeof(double)*n) );
    check( cudaMalloc((void**)&d_out, sizeof(double)*nblks) );

    generateRandomNumbers(n, d_Z);

    kernelQuestion6<<<nblks, nthds, sizeof(double)*nthds>>>(S0, r, T, sigma, K, n, d_Z, d_out);
    check( cudaGetLastError() );

    double *out = new double[nblks];
    check( cudaMemcpy(out, d_out, sizeof(double)*nblks, cudaMemcpyDeviceToHost) );
    double sum = 0.0;
    for(int i=0; i<nblks; i++) {
        sum += out[i];
    }

    cout << "average=" << sum/n << endl;

    check( cudaFree(d_Z) );
    check( cudaFree(d_out) );
    delete[] out;
}

#endif
