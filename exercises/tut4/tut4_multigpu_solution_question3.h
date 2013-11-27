

__global__ void kernelQuestion3(int gpuId, int n, unsigned char * d_data)
{
    if(threadIdx.x==0) {
        for(int i=0; i<n; i++)
            printf("GPU %d sees d_data[%d]=%d\n", gpuId, i, d_data[i]);
    }
}


void question3()
{
    cout << endl << endl;
    cout << "Tutorial 4, Question 3" << endl;
    cout.precision(14);

    int gpu0, gpu1;
    getPeerAccessDevices(gpu0, gpu1);

    int n = 5;
    // Our one GPU memory array
    unsigned char * d_data = NULL;
    check( cudaSetDevice(gpu0) );
    check( cudaMalloc((void**)&d_data, sizeof(unsigned char)*n) );
    check( cudaMemset(d_data, 23, sizeof(unsigned char)*n) );

    // Print out from GPU 0
    cout << "Printing from GPU " << gpu0 << " ..." << endl;
    kernelQuestion3<<<1,1>>>(gpu0, n, d_data);
    check( cudaDeviceSynchronize() );

    cout << endl << "Now trying to print from GPU " << gpu1 << " ..." << endl;

    // Print out from GPU 1
    check( cudaSetDevice(gpu1) );
    check( cudaDeviceEnablePeerAccess(gpu0, 0) );
    kernelQuestion3<<<1,1>>>(gpu1, n, d_data);
    check( cudaDeviceSynchronize() );
    check( cudaDeviceDisablePeerAccess(gpu0) );

    check( cudaFree(d_data) );
}

