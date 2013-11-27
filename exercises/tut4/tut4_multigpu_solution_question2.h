

__device__ float d_blkSum = 0.0f;
__global__
void kernelQuestion2(
    float S0,
    float r,
    float T,
    float sigma,
    float K,
    int n,
    float *d_Z)
{
    int thdsPerBlk = blockDim.x;	// Number of threads per block
    int tidxKernel = blockIdx.x*thdsPerBlk + threadIdx.x;
    int totalThds = thdsPerBlk*gridDim.x;

    extern __shared__ float partials[];

    float sum = 0.0f;
    float sqT = sqrt(T);
    for(int zIdx=tidxKernel; zIdx<n; zIdx+=totalThds) {
        float ST = S0*exp( (r-0.5f*sigma*sigma)*T + sigma*sqT*d_Z[zIdx] );
        float x = (ST>K ? ST-K : 0.0f);
        x = exp(-r*T)*x;
        sum += x;
    }

    partials[threadIdx.x] = sum;
    __syncthreads();

    if(threadIdx.x == 0) {
        sum = 0.0;
        for(int i=0; i<thdsPerBlk; i++) {
            sum += partials[i];
        }
        atomicAdd(&d_blkSum, sum);
    }
}


void question2()
{
    cout << endl << endl;
    cout << "Tutorial 4, Question 2" << endl;
    cout.precision(14);

    int ngpus;
    check( cudaGetDeviceCount(&ngpus) );
    cout << "We have " << ngpus << " CUDA capable devices" << endl << endl;;
    if(ngpus < 2) {
        cout << "Error: we don't have 2 GPUs!!" << endl << endl;
        exit(-1);
    }
    ngpus = 2;
    const int gpu0 = 0, gpu1 = 1;

    // Constants for the Monte Carlo
    float r = 0.02f, T = 1.0f, sigma = 0.09f, K = 100.0f, S0 = 100.0f;
    // Problem size
    const int n = 100000000;
    const int nthds = 192;
    const int nblks = 224;


    /*
     * Run full problem on one GPU and get runtime
     */

    // Storage for each GPU
    float *d0_Z;
    // Allocate storage and generate random numbers
    check( cudaSetDevice(gpu0) );
    check( cudaMalloc((void**)&d0_Z, sizeof(float)*n) );
    generateRandomNumbers(n, 0, d0_Z);
    // Launch kernel and measure runtime
    double start = omp_get_wtime();
    kernelQuestion2<<<nblks, nthds, sizeof(float)*nthds>>>(S0, r, T, sigma, K, n, d0_Z);
    check( cudaGetLastError() );
    // Now synchronize and copy back
    float sum;
    check( cudaMemcpyFromSymbol(&sum, d_blkSum, sizeof(float)) );
    double stop = omp_get_wtime();
    double time = (stop-start)*1000.0;
    cout << "Full problem on 1 GPU: " << endl;
    cout << "   average = " << sum/n << endl;
    cout << "   runtime = " << time << "ms" << endl;


    // Zero out the memory
    sum = 0.0f;
    check( cudaMemcpyToSymbol(d_blkSum, &sum, sizeof(float)) );

    /*
     * Now run same problem on two GPUs
     */

    // Problem sizes for each GPU
    int n0, n1;
    n0 = n/ngpus;
    n1 = n - n0;
    check( cudaSetDevice(gpu1) );
    // Allocate memory on GPU1 and generate random numbers (offset into sequence!)
    float *d1_Z;
    check( cudaMalloc((void**)&d1_Z, sizeof(float)*n1) );
    generateRandomNumbers(n1, n0, d1_Z);

    start = omp_get_wtime();
    // Launch kernel on GPU 0
    check( cudaSetDevice(gpu0) );
    kernelQuestion2<<<nblks, nthds, sizeof(float)*nthds>>>(S0, r, T, sigma, K, n0, d0_Z);
    check( cudaGetLastError() );
    // Launch kernel on GPU 0
    check( cudaSetDevice(gpu1) );
    kernelQuestion2<<<nblks, nthds, sizeof(float)*nthds>>>(S0, r, T, sigma, K, n1, d1_Z);
    check( cudaGetLastError() );
    float blkSum[2];
    // Now synchronize and copy back
    check( cudaSetDevice(gpu0) );
    check( cudaMemcpyFromSymbol(&blkSum[0], d_blkSum, sizeof(float)) );
    check( cudaSetDevice(gpu1) );
    check( cudaMemcpyFromSymbol(&blkSum[1], d_blkSum, sizeof(float)) );
    // Do final reduction
    sum = blkSum[0] + blkSum[1];

    stop = omp_get_wtime();
    time = (stop-start)*1000.0;

    cout << "Problem split across 2 GPUs: " << endl;
    cout << "   average = " << sum/n << endl;
    cout << "   runtime = " << time << "ms" << endl;

    check( cudaFree(d0_Z) );
    check( cudaFree(d1_Z) );
}

