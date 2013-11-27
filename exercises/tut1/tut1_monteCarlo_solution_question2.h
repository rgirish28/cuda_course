#ifndef _TUTORIAL1_QUESTION2_H
#define _TUTORIAL1_QUESTION2_H



__global__ void kernelQuestion2(
    int n,
    double * d_ans,
    double *d_vec1,
    double *d_vec2)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if(tid < n) {
        d_ans[tid] = d_vec1[tid] + d_vec2[tid];
    }
}

void question2()
{
    cout << endl;
    cout << "Tutorial 1, Question 2" << endl;

    int n = 2013;
    int nthds = 256;
    int nblks = 13;

    double *vec1 = new double[n];
    double *vec2 = new double[n];
    double *ans = new double[n];
    getVectorsToAdd(n, vec1, vec2);

    double *d_ans = NULL, *d_vec1 = NULL, *d_vec2 = NULL;
    check( cudaMalloc((void**)&d_ans, sizeof(double)*n) );
    check( cudaMalloc((void**)&d_vec1, sizeof(double)*n) );
    check( cudaMalloc((void**)&d_vec2, sizeof(double)*n) );
    check( cudaMemcpy(d_vec1, vec1, sizeof(double)*n, cudaMemcpyHostToDevice) );
    check( cudaMemcpy(d_vec2, vec2, sizeof(double)*n, cudaMemcpyHostToDevice) );

    kernelQuestion2<<<nblks, nthds>>>(n, d_ans, d_vec1, d_vec2);
    check( cudaGetLastError() );

    check( cudaMemcpy(ans, d_ans, sizeof(double)*n, cudaMemcpyDeviceToHost) );
    for(int i=0; i<n; i++) {
        if(ans[i]!=100) {
            cout << "Error: ans[" << i << "]=" << ans[i] << endl;
            exit(-1);
        }
    }
    cout << "All values in ans equal 100!" << endl;

    delete[] ans;
    delete[] vec1;
    delete[] vec2;
    check( cudaFree(d_ans) );
    check( cudaFree(d_vec1) );
    check( cudaFree(d_vec2) );
}


#endif
