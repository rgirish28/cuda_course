#ifndef _TUTORIAL3_QUESTION6_H
#define _TUTORIAL3_QUESTION6_H


__global__ void kernelQuestion6(int n, float2 * d_dest, float2 * d_src)
{
    for(int idx=threadIdx.x; idx<n; idx+=blockDim.x) {
        float2 x = d_src[idx];
        d_dest[idx] = x;
    }
}



void question6()
{
    cout << endl;
    cout << "Tutorial 2, Question 6" << endl;

    const int n = 67108864/2;

    float2 *d_dest = NULL;
    float2 *d_src = NULL;

    check( cudaMalloc((void**)&d_dest, sizeof(float2)*n) );
    check( cudaMalloc((void**)&d_src, sizeof(float2)*n) );

    autoTuneMemcpyKernel(n, d_dest, d_src);

    check( cudaFree(d_dest) );
    check( cudaFree(d_src) );

}



#endif
