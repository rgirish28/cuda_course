#ifndef _TUTORIAL1_QUESTION4_H
#define _TUTORIAL1_QUESTION4_H



__global__ void kernelQuestion4(
    int n,
    double *d_Z,
    double *d_out)
{
    extern __shared__ double partials[];

    double sum = 0.0;
    for(int i=threadIdx.x; i<n; i+=blockDim.x) {
        sum += d_Z[i];
    }
    partials[threadIdx.x] = sum;
    __syncthreads();

    if(threadIdx.x==0) {
        sum = 0.0;
        for(int i=0; i<blockDim.x; i++) {
            sum += partials[i];
        }

        d_out[blockIdx.x] = sum;
    }
}

void question4()
{
    cout << endl;
    cout << "Tutorial 1, Question 4" << endl;

    int n = 100000;
    int nthds = 448;

    double *d_Z = NULL;
    double *d_out = NULL;

    check( cudaMalloc((void**)&d_Z, sizeof(double)*n) );
    check( cudaMalloc((void**)&d_out, sizeof(double)) );

    generateRandomNumbers(n, d_Z);

    kernelQuestion4<<<1, nthds, sizeof(double)*nthds>>>(n, d_Z, d_out);
    check( cudaGetLastError() );

    double out;
    check( cudaMemcpy(&out, d_out, sizeof(double), cudaMemcpyDeviceToHost) );

    cout << "sum=" << out << endl << endl;

    check( cudaFree(d_Z) );
    check( cudaFree(d_out) );
}


#endif
