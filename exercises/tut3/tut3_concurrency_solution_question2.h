#ifndef _TUTORIAL2_QUESTION2_H
#define _TUTORIAL2_QUESTION2_H



void question2(
    int nRows,
    int nCols,
    int midDim,
    int nMatrices
)
{
    cout << endl;
    cout << "Tutorial 3, Question 2:" << endl;

    double *A = new double[nRows*midDim];
    double *Bs = NULL;
    double *Cs = NULL;

    // Allocate pinned memory
    check( cudaHostAlloc((void**)&Bs, sizeof(double)*midDim*nCols*nMatrices, cudaHostAllocMapped) );
    check( cudaHostAlloc((void**)&Cs, sizeof(double)*nRows*nCols*nMatrices, cudaHostAllocMapped) );
    // Populate the matrices
    makeAMatrix(A, nRows, midDim);
    makeBMatrices(Bs, midDim, nCols, nMatrices);


    double *d_A, *d_Bs, *d_Cs;
    check( cudaMalloc((void**)&d_A, sizeof(double)*nRows*midDim) );
    check( cudaHostGetDevicePointer((void**)&d_Bs, Bs, 0) );
    check( cudaHostGetDevicePointer((void**)&d_Cs, Cs, 0) );

    // Create timing events
    cudaEvent_t start, stop;
    check( cudaEventCreate(&start) );
    check( cudaEventCreate(&stop) );

    check( cudaEventRecord(start, 0) );

    // Copy A only once
    check( cudaMemcpy(d_A, A, sizeof(double)*nRows*midDim, cudaMemcpyHostToDevice) );
    for(int i=0; i<nMatrices; i++) {
        launchKernel(nRows, nCols, midDim, d_A, &d_Bs[i*midDim*nCols], &d_Cs[i*nRows*nCols],0);
    }
    check( cudaEventRecord(stop, 0) );

    check( cudaDeviceSynchronize() );
    float time;
    check( cudaEventElapsedTime(&time, start, stop) );
    cout << "Did " << nMatrices << " matrix multiplications and copies in " << time << "ms " << endl;

    // Check results
    checkCs(nRows, nCols, midDim, nMatrices, Cs);

    delete[] A;
    check( cudaFreeHost(Bs) );
    check( cudaFreeHost(Cs) );

    check( cudaFree(d_A) );
    check( cudaEventDestroy(start) );
    check( cudaEventDestroy(stop) );
}


#endif
