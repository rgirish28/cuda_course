#ifndef _TUTORIAL2_QUESTION4_H
#define _TUTORIAL2_QUESTION4_H


void question4(
    int nRows,
    int nCols,
    int midDim,
    int nMatrices
)
{
    cout << endl;
    cout << "Concurrent Copy Question 4:" << endl;

    double *A = new double[nRows*midDim];
    double *Bs = new double[midDim*nCols*nMatrices];
    double *Cs = new double[nRows*nCols*nMatrices];

    double *d_A;
    // Need 3 B buffers and 3 C buffers and 3 staging buffers, since three streams
    double *d_B[3], *d_C[3];
    double *stageUpld[3];
    double *stageDownld[3];

    // Allocate memory
    check( cudaMalloc((void**)&d_A, sizeof(double)*nRows*midDim) );
    for(int i=0; i<3; i++) {
        check( cudaMalloc((void**)&d_B[i], sizeof(double)*midDim*nCols) );
        check( cudaMalloc((void**)&d_C[i], sizeof(double)*nRows*nCols) );
        check( cudaHostAlloc((void**)&stageUpld[i], sizeof(double)*midDim*nCols, cudaHostAllocWriteCombined) );
        check( cudaHostAlloc((void**)&stageDownld[i], sizeof(double)*nRows*nCols, cudaHostAllocDefault) );
    }

    makeAMatrix(A, nRows, midDim);
    makeBMatrices(Bs, midDim, nCols, nMatrices);

    // Time the various legs
    timeH2DCopyForA(d_A, A, nRows*midDim);
    timeH2HCopyForStageUpld(stageUpld[0], Bs, midDim*nCols);
    timeH2DD2HCopyForBC(stageUpld[0], d_B[0], midDim*nCols, stageDownld[0], d_C[0], nRows*nCols);
    timeKernel(d_A, d_B[0], d_C[0], nRows, nCols, midDim);
    timeH2HCopyForStageDownld(Cs, stageDownld[0], nRows*nCols);

    // Create streams
    cudaStream_t streams[3];
    // Create timing events
    cudaEvent_t start, stop;
    cudaEvent_t jobComplete[3];

    check( cudaEventCreate(&start) );
    check( cudaEventCreate(&stop) );
    for(int i=0; i<3; i++) {
        check( cudaStreamCreate(&streams[i]) );
        check( cudaEventCreateWithFlags(&jobComplete[i], cudaEventDisableTiming) );
    }


    float time;
    check( cudaEventRecord(start, 0) );

    // Copy A only once
    check( cudaMemcpy(d_A, A, sizeof(double)*nRows*midDim, cudaMemcpyHostToDevice) );
    int stmIdx = 0, i = 0;

    // Start first stream off
    memcpy(stageUpld[stmIdx], &Bs[i*midDim*nCols], sizeof(double)*midDim*nCols);
    check( cudaMemcpyAsync(d_B[stmIdx], stageUpld[stmIdx], sizeof(double)*midDim*nCols, cudaMemcpyHostToDevice, streams[stmIdx]) );
    launchKernel(nRows, nCols, midDim, d_A, d_B[stmIdx], d_C[stmIdx], streams[stmIdx]);
    check( cudaMemcpyAsync(stageDownld[stmIdx], d_C[stmIdx], sizeof(double)*nRows*nCols, cudaMemcpyDeviceToHost, streams[stmIdx]) );
    // Need to wait for this job to complete before we can copy from stageDownld to Cs.
    check( cudaEventRecord(jobComplete[stmIdx], streams[stmIdx]) );
    stmIdx++;
    i++;

    // Now do remaining jobs
    for(; i<nMatrices; i++) {
        // Enqueue next job while waiting for previous job to complete
        memcpy(stageUpld[stmIdx], &Bs[i*midDim*nCols], sizeof(double)*midDim*nCols);
        check( cudaMemcpyAsync(d_B[stmIdx], stageUpld[stmIdx], sizeof(double)*midDim*nCols, cudaMemcpyHostToDevice, streams[stmIdx]) );
        launchKernel(nRows, nCols, midDim, d_A, d_B[stmIdx], d_C[stmIdx], streams[stmIdx]);
        check( cudaMemcpyAsync(stageDownld[stmIdx], d_C[stmIdx], sizeof(double)*nRows*nCols, cudaMemcpyDeviceToHost, streams[stmIdx]) );
        check( cudaEventRecord(jobComplete[stmIdx], streams[stmIdx]) );

        // What was previous stream?
        int prevStm = (stmIdx-1<0 ? 2 : stmIdx-1);
        // Wait for job enqueued on previous stream to complete
        check( cudaEventSynchronize(jobComplete[prevStm]) );
        // Copy previous results from download staging buffer into Cs
        memcpy(&Cs[(i-1)*nRows*nCols], stageDownld[prevStm], sizeof(double)*nRows*nCols);

        stmIdx++;
        stmIdx %= 3;
    }
    // Copy data results data from final iteration
    // What was previous stream?
    int prevStm = (stmIdx-1<0 ? 2 : stmIdx-1);
    check( cudaEventSynchronize(jobComplete[prevStm]) );
    memcpy(&Cs[(i-1)*nRows*nCols], stageDownld[prevStm], sizeof(double)*nRows*nCols);

    check( cudaEventRecord(stop, 0) );

    check( cudaDeviceSynchronize() );
    check( cudaEventElapsedTime(&time, start, stop) );
    cout << "Did " << nMatrices << " matrix multiplications and copies in " << time << "ms " << endl;

    // Check results
    checkCs(nRows, nCols, midDim, nMatrices, Cs);

    delete[] A;
    delete[] Bs;
    delete[] Cs;
    for(int i=0; i<3; i++) {
        check( cudaFreeHost(stageUpld[i]) );
        check( cudaFreeHost(stageDownld[i]) );
        check( cudaFree(d_B[i]) );
        check( cudaFree(d_C[i]) );
        check( cudaStreamDestroy(streams[i]) );
    }

    check( cudaFree(d_A) );
    check( cudaEventDestroy(start) );
    check( cudaEventDestroy(stop) );
}


#endif
