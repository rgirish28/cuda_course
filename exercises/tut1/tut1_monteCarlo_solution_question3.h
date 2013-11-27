#ifndef _TUTORIAL1_QUESTION3_H
#define _TUTORIAL1_QUESTION3_H


__global__ void kernelQuestion3(
    int n,
    double *d_Z,
    double *d_out)
{
    double sum = 0.0;
    for(int i=threadIdx.x; i<n; i+=blockDim.x) {
        sum += d_Z[i];
    }

    d_out[threadIdx.x] = sum;
}

void question3()
{
    cout << endl;
    cout << "Tutorial 1, Question 3" << endl;

    int n = 100000;
    int nthds = 448;

    double *d_Z = NULL;
    double *d_out = NULL;


    check( cudaMalloc((void**)&d_Z, sizeof(double)*n) );
    check( cudaMalloc((void**)&d_out, sizeof(double)*nthds) );

    generateRandomNumbers(n, d_Z);

    kernelQuestion3<<<1, nthds>>>(n, d_Z, d_out);
    check( cudaGetLastError() );

    double * out = new double[nthds];
    check( cudaMemcpy(out, d_out, sizeof(double)*nthds, cudaMemcpyDeviceToHost) );

    double sum = 0.0;
    for(int i=0; i<nthds; i++) {
        sum += out[i];
    }
    cout << "sum=" << sum << endl << endl;

    check( cudaFree(d_Z) );
    check( cudaFree(d_out) );
    delete[] out;
}


#endif
