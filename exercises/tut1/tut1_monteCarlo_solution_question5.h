#ifndef _TUTORIAL1_QUESTION5_H
#define _TUTORIAL1_QUESTION5_H



__global__ void kernelQuestion5(
    int n,
    double *d_Z,
    double *d_out)
{
    extern __shared__ double partials[];
    int tidxKernel = blockIdx.x*blockDim.x + threadIdx.x;
    const int totalThds = gridDim.x*blockDim.x;

    double sum = 0.0;
    for(int zIdx=tidxKernel; zIdx<n; zIdx += totalThds) {
        sum += d_Z[zIdx];
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

void question5()
{
    cout << endl;
    cout << "Tutorial 1, Question 5" << endl;

    int n = 100000;
    int nthds = 448;
    int nblks = 20;

    double *d_Z = NULL;
    double *d_out = NULL;

    check( cudaMalloc((void**)&d_Z, sizeof(double)*n) );
    check( cudaMalloc((void**)&d_out, sizeof(double)*nblks) );

    generateRandomNumbers(n, d_Z);

    kernelQuestion5<<<nblks, nthds, sizeof(double)*nthds>>>(n, d_Z, d_out);
    check( cudaGetLastError() );

    double *out = new double[nblks];
    check( cudaMemcpy(out, d_out, sizeof(double)*nblks, cudaMemcpyDeviceToHost) );
    double sum = 0.0;
    for(int i=0; i<nblks; i++) {
        sum += out[i];
    }

    cout << "sum=" << sum << endl << endl;

    check( cudaFree(d_Z) );
    check( cudaFree(d_out) );
    delete[] out;
}


#endif
