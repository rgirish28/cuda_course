
__global__ void kernelQuestion4(int n, float4 * d_dest, float4 * d_src)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int nthds = gridDim.x*blockDim.x;

    for(int idx=tid; idx<n; idx+=nthds) {
        float4 x = d_src[idx];
        d_dest[idx] = x;
    }
}


void question4()
{
    cout << endl << endl;
    cout << "Tutorial 4, Question 4" << endl;

    const int n = 445644800/4;
    float4 *d_dest = NULL;
    float4 *d_src = NULL;
    const double szInGB = n*sizeof(float4)/1073741824.0f;

    /*
     * Convince yourself that under UAS, cudaMemcpy will work
     * regardless of source, destination or active GPU
     */
    // Copy from GPU0 to GPU1 when GPU2 is active
    check( cudaSetDevice(0) );
    check( cudaMalloc((void**)&d_src, sizeof(float4)*n) );
    check( cudaSetDevice(1) );
    check( cudaMalloc((void**)&d_dest, sizeof(float4)*n) );

    cout << "Setting GPU2 active ..." << endl;
    check (cudaSetDevice(2) );
    for(int i=0; i<5; i++) {
        double start = omp_get_wtime();
        check( cudaMemcpy(d_dest, d_src, sizeof(float4)*n, cudaMemcpyDefault) );
        double stop = omp_get_wtime();
        double time = stop - start;
        cout << "cudaMemcpy GPU0->GPU1 (not peer): copied data at " << szInGB/time << "GB/s" << endl;
    }
    // Free memory - UAS again makes this easier (normally
    // these calls would create errors since GPU2 is active)
    check( cudaFree(d_src) );
    check( cudaFree(d_dest) );

    cout << endl;

    /*
     * Now investigate peer-to-peer direct copy
     */
    int gpu0, gpu1;
    getPeerAccessDevices(gpu0, gpu1);
    check( cudaSetDevice(gpu0) );
    check( cudaMalloc((void**)&d_src, sizeof(float4)*n) );
    cout << "Setting GPU" << gpu1 << " active ..." << endl;
    check( cudaSetDevice(gpu1) );
    check( cudaMalloc((void**)&d_dest, sizeof(float4)*n) );
    // Call cudaMemcpy without enabling peer access
    for(int i=0; i<5; i++) {
        double start = omp_get_wtime();
        check( cudaMemcpy(d_dest, d_src, sizeof(float4)*n, cudaMemcpyDefault) );
        double stop = omp_get_wtime();
        double time = stop - start;
        cout << "cudaMemcpy GPU" << gpu0 << "->GPU" << gpu1 << " (peer access not enabled): "
             "copied data at " << szInGB/time << "GB/s" << endl;
    }
    // Now enable peer access
    cout << "Enabling GPU" << gpu1 << " to access GPU" << gpu0 << "'s memory directly ..." << endl;
    check( cudaDeviceEnablePeerAccess(gpu0, 0) );
    // The currently active context makes no difference
    // check( cudaSetDevice(6) );
    for(int i=0; i<5; i++) {
        double start = omp_get_wtime();
        check( cudaMemcpy(d_dest, d_src, sizeof(float4)*n, cudaMemcpyDefault) );
        check( cudaDeviceSynchronize() );
        double stop = omp_get_wtime();
        double time = stop - start;
        cout << "GPU" << gpu0 << "->GPU" << gpu1 << " direct copy : copied data at " << szInGB/time << "GB/s" << endl;
    }
    // Swap the direction of copy
    cout << "Swapping direction of copy ..." << endl;
    for(int i=0; i<5; i++) {
        double start = omp_get_wtime();
        check( cudaMemcpy(d_src, d_dest, sizeof(float4)*n, cudaMemcpyDefault) );
        check( cudaDeviceSynchronize() );
        double stop = omp_get_wtime();
        double time = stop - start;
        cout << "GPU" << gpu1 << "->GPU" << gpu0 << " direct copy : copied data at " << szInGB/time << "GB/s" << endl;
    }


    /*
     * Now see how fast a kernel can access memory on a peer
     */
    cout << "Setting GPU" << gpu1 << " active ..." << endl;
    check( cudaSetDevice(gpu1) );
    for(int i=0; i<5; i++) {
        double start = omp_get_wtime();
        kernelQuestion4<<<25, 1024>>>(n, d_dest, d_src);
        check( cudaDeviceSynchronize() );
        double stop = omp_get_wtime();
        double time = stop - start;
        cout << "GPU" << gpu0 << "->GPU" << gpu1 << " copy by kernel: copied data at " << szInGB/time << "GB/s" << endl;
    }
    cout << "Swapping direction of copy ..." << endl;
    // Swap the direction of copy
    for(int i=0; i<5; i++) {
        double start = omp_get_wtime();
        kernelQuestion4<<<25, 1024>>>(n, d_src, d_dest);
        check( cudaDeviceSynchronize() );
        double stop = omp_get_wtime();
        double time = stop - start;
        cout << "GPU" << gpu1 << "->GPU" << gpu0 << " copy by kernel: copied data at " << szInGB/time << "GB/s" << endl;
    }
    check( cudaDeviceDisablePeerAccess(gpu0) );
    // Now try and enable bidirectional peer access and see if there's any difference
    cout << "Enabling bidirectional peer access ... " << endl;
    check( cudaSetDevice(gpu0) );
    check( cudaDeviceEnablePeerAccess(gpu1, 0) );
    check( cudaSetDevice(gpu1) );
    check( cudaDeviceEnablePeerAccess(gpu0, 0) );
    cout << "Setting GPU" << gpu1 << " active ..." << endl;
    check( cudaSetDevice(gpu1) );
    for(int i=0; i<5; i++) {
        double start = omp_get_wtime();
        kernelQuestion4<<<25, 1024>>>(n, d_dest, d_src);
        check( cudaDeviceSynchronize() );
        double stop = omp_get_wtime();
        double time = stop - start;
        cout << "GPU" << gpu0 << "->GPU" << gpu1 << " copy by kernel: copied data at " << szInGB/time << "GB/s" << endl;
    }
    cout << "Swapping direction of copy ..." << endl;
    // Swap the direction of copy
    for(int i=0; i<5; i++) {
        double start = omp_get_wtime();
        kernelQuestion4<<<25, 1024>>>(n, d_src, d_dest);
        check( cudaDeviceSynchronize() );
        double stop = omp_get_wtime();
        double time = stop - start;
        cout << "GPU" << gpu1 << "->GPU" << gpu0 << " copy by kernel: copied data at " << szInGB/time << "GB/s" << endl;
    }

    check( cudaDeviceDisablePeerAccess(gpu0) );
    check( cudaSetDevice(gpu0) );
    check( cudaDeviceDisablePeerAccess(gpu1) );
    check( cudaFree(d_dest) );
    check( cudaFree(d_src) );
}


