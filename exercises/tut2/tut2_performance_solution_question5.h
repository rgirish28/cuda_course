#ifndef _TUTORIAL3_QUESTION5_H
#define _TUTORIAL3_QUESTION5_H



__global__ void kernelQuestion5(int n, float * d_dest, float * d_src)
{
    for(int idx=threadIdx.x; idx<n; idx+=blockDim.x) {
        float x = d_src[idx];
        d_dest[idx] = x;
    }
}


void question5()
{
    cout << endl;
    cout << "Tutorial 2, Question 5" << endl;

    const int n = 67108864;

    float *d_dest = NULL;
    float *d_src = NULL;

    check( cudaMalloc((void**)&d_dest, sizeof(float)*n) );
    check( cudaMalloc((void**)&d_src, sizeof(float)*n) );

    autoTuneMemcpyKernel(n, d_dest, d_src);

    check( cudaFree(d_dest) );
    check( cudaFree(d_src) );
}




#endif