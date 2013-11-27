#ifndef _TUTORIAL3_QUESTION7_H
#define _TUTORIAL3_QUESTION7_H


__global__ void kernelQuestion7(int n, float4 * d_dest, float4 * d_src)
{
    for(int idx=threadIdx.x; idx<n; idx+=blockDim.x) {
        float4 x = d_src[idx];
        d_dest[idx] = x;
    }
}



void question7()
{
    cout << endl;
    cout << "Tutorial 2, Question 7" << endl;

    const int n = 67108864/4;

    float4 *d_dest = NULL;
    float4 *d_src = NULL;

    check( cudaMalloc((void**)&d_dest, sizeof(float4)*n) );
    check( cudaMalloc((void**)&d_src, sizeof(float4)*n) );

    autoTuneMemcpyKernel(n, d_dest, d_src);

    check( cudaFree(d_dest) );
    check( cudaFree(d_src) );

}




#endif