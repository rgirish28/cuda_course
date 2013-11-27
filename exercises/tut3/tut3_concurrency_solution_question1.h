#ifndef _TUTORIAL2_QUESTION1_H
#define _TUTORIAL2_QUESTION1_H


void question1(
    int nRows,
    int nCols,
    int midDim,
    int nMatrices
)
{
    cout << endl;
    cout << "Tutorial 3, Question 1:" << endl;

    double *A = new double[nRows*midDim];
    double *Bs = new double[midDim*nCols*nMatrices];
    double * Cs = new double[nRows*nCols*nMatrices];

    makeAMatrix(A, nRows, midDim);
    makeBMatrices(Bs, midDim, nCols, nMatrices);

    double *d_A, *d_B, *d_C;
    check( cudaMalloc((void**)&d_A, sizeof(double)*nRows*midDim) );
    check( cudaMalloc((void**)&d_B, sizeof(double)*midDim*nCols) );
    check( cudaMalloc((void**)&d_C, sizeof(double)*nRows*nCols) );

    // Timing functions
    timeH2DCopyForA(d_A, A, nRows*midDim);
    timeH2DCopyForB(d_B, Bs, midDim*nCols);
    timeD2HCopyForC(Cs, d_C, nRows*nCols);
    timeKernel(d_A, d_B, d_C, nRows, nCols, midDim);

    cudaEvent_t start, stop;
    check( cudaEventCreate(&start) );
    check( cudaEventCreate(&stop) );
    float time;


    // Now we start the proper run
    check( cudaEventRecord(start, 0) );

    // Copy A only once
    check( cudaMemcpy(d_A, A, sizeof(double)*nRows*midDim, cudaMemcpyHostToDevice) );

    for(int i=0; i<nMatrices; i++) {
        check( cudaMemcpy(d_B, &Bs[i*midDim*nCols], sizeof(double)*midDim*nCols, cudaMemcpyHostToDevice) );
        launchKernel(nRows, nCols, midDim, d_A, d_B, d_C, 0);
        check( cudaMemcpy(&Cs[i*nRows*nCols], d_C, sizeof(double)*nRows*nCols, cudaMemcpyDeviceToHost) );
    }
    check( cudaEventRecord(stop, 0) );

    check( cudaDeviceSynchronize() );		// Not really necessary since everything is in serial anyway
    check( cudaEventElapsedTime(&time, start, stop) );
    cout << "Did " << nMatrices << " matrix multiplications and copies in " << time << "ms " << endl;


    checkCs(nRows, nCols, midDim, nMatrices, Cs);
    delete[] A;
    delete[] Bs;
    delete[] Cs;

    check( cudaFree(d_A) );
    check( cudaFree(d_B) );
    check( cudaFree(d_C) );
    check( cudaEventDestroy(start) );
    check( cudaEventDestroy(stop) );
}


#endif
