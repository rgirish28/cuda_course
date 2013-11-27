#ifndef _TUTORIAL3_QUESTION8_H
#define _TUTORIAL3_QUESTION8_H

__global__ void kernelQuestion8(int n, float * d_dest, float * d_src)
{
    const int PREF = 10;
    float pref[PREF];

    int ridx = threadIdx.x;
    for(int i=0; i<PREF; i++) {
        pref[i] = d_src[ridx];
        ridx += blockDim.x;
    }
    int widx = threadIdx.x;
    while(widx<n) {
#pragma unroll 10
        for(int i=0; i<PREF; i++) {
            float x = pref[i];
            pref[i] = d_src[ridx];
            ridx += blockDim.x;
            d_dest[widx] = x;
            widx += blockDim.x;
        }
    }
}


void question8()
{
    cout << endl;
    cout << "Tutorial 2, Question 8" << endl;

    const int n = 67108864;

    float *d_dest = NULL;
    float *d_src = NULL;

    check( cudaMalloc((void**)&d_dest, sizeof(float)*(n+1024*16)) );
    check( cudaMalloc((void**)&d_src, sizeof(float)*(n+1024*16)) );

    autoTuneMemcpyKernel(n, d_dest, d_src);

    check( cudaFree(d_dest) );
    check( cudaFree(d_src) );

}





#endif
