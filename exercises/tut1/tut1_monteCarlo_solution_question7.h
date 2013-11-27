#ifndef _TUTORIAL1_QUESTION7_H
#define _TUTORIAL1_QUESTION7_H


__global__ void kernelQuestion7(
    double S0,
    double r,
    double T,
    double sigma,
    double K,
    int n,
    double *d_Z,
    double *d_out,
    int M,
    double *d_sumsq)
{
    int thdsPerBlk = blockDim.x;	// Number of threads per block
    int tidxKernel = blockIdx.x*thdsPerBlk + threadIdx.x;
    int totalThds = thdsPerBlk*gridDim.x;

    extern __shared__ double partials[];

    double sum = 0.0;
    double sumsq = 0.0;
    double sqT = sqrt(T);

    if(blockIdx.x < M) {
        for(int zIdx=tidxKernel; zIdx<n; zIdx+=totalThds) {
            double ST = S0*exp( (r-0.5*sigma*sigma)*T + sigma*sqT*d_Z[zIdx] );
            double x = (ST>K ? ST-K : 0.0);
            x = exp(-r*T)*x;
            sum += x;
            sumsq += x*x;
        }
    } else {
        for(int zIdx=tidxKernel; zIdx<n; zIdx+=totalThds) {
            double ST = S0*exp( (r-0.5*sigma*sigma)*T + sigma*sqT*d_Z[zIdx] );
            double x = (ST>K ? ST-K : 0.0);
            x = exp(-r*T)*x;
            sum += x;
        }
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

    // Hold the other threads so that thread0 can do sum
    __syncthreads();
    // Now write sum of squares
    partials[threadIdx.x] = sumsq;
    __syncthreads();

    if(blockIdx.x < M && threadIdx.x == 0) {
        sum = 0.0;
        for(int i=0; i<thdsPerBlk; i++) {
            sum += partials[i];
        }
        d_sumsq[blockIdx.x] = sum;
    }
}

void question7()
{
    cout << endl;
    cout << "Tutorial 1, Question 7" << endl;

    int n = 100000000;
    int nthds = 672;
    int nblks = 140;
    int M = 35;

    double
    r = 0.02,
    T = 1.0,
    sigma = 0.09,
    K = 100.0,
    S0 = 100.0;

    double *d_Z = NULL;
    double *d_out = NULL;
    double *d_sumsq = NULL;

    check( cudaMalloc((void**)&d_Z, sizeof(double)*n) );
    check( cudaMalloc((void**)&d_out, sizeof(double)*nblks) );
    check( cudaMalloc((void**)&d_sumsq, sizeof(double)*nblks) );

    generateRandomNumbers(n, d_Z);

    kernelQuestion7<<<nblks, nthds, sizeof(double)*nthds>>>(S0, r, T, sigma, K, n, d_Z, d_out, M, d_sumsq);
    check( cudaGetLastError() );

    double *out = new double[nblks];
    double * sumsq = new double[M];
    check( cudaMemcpy(out, d_out, sizeof(double)*nblks, cudaMemcpyDeviceToHost) );
    check( cudaMemcpy(sumsq, d_sumsq, sizeof(double)*M, cudaMemcpyDeviceToHost) );

    double sum = 0.0;
    for(int i=0; i<nblks; i++) {
        sum += out[i];
    }
    double mcEst = sum/n;
    cout << "average=" << mcEst << endl;
    cout << endl;


    sum = 0.0;
    double sq = 0.0;
    for(int i=0; i<M; i++) {
        sum += out[i];
        sq += sumsq[i];
    }
    // Number of loops that all blocks will do (note: round down)
    int nloops = n/(nthds*nblks);
    // Remainder of points to be cleaned up on final loop
    int rem = n%(nthds*nblks);
    // Number of points read by first M thread blocks
    int points = M*nthds*nloops + min(rem,M*nthds);
    double ave = sum/points;
    double variance = sq/points - ave*ave;
    double stdev = sqrt(variance);
    cout << "Sum of first M blocks=" << sum << endl;
    cout << "Sum of squares of first M blocks=" << sq << endl;
    cout << "Variance of first M blocks=" << variance << endl;
    cout << "  99% Confidence interval is (" << mcEst - 3.0*stdev/sqrt((double)n) << ", " << mcEst + 3.0*stdev/sqrt((double)n) << ")" << endl;
    cout << "  Width of interval is " << 6.0*stdev/sqrt((double)n) << endl << endl;

    check( cudaFree(d_Z) );
    check( cudaFree(d_out) );
    check( cudaFree(d_sumsq) );
    delete[] sumsq;
    delete[] out;
}


#endif
