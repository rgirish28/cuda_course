

void __global__ helloWorld(int devId)
{
    if(threadIdx.x==0 && blockIdx.x==0) {
        printf("Hello World from Device %d\n",devId);
    }
}


void question1()
{
    cout << endl << endl;
    cout << "Tutorial 4, Question 1" << endl;

    int ngpus;
    check( cudaGetDeviceCount(&ngpus) );
    cout << "We have " << ngpus << " CUDA capable devices" << endl;

    for(int i=0; i<ngpus; i++) {
        cout << endl;
        cudaDeviceProp prop;
        check( cudaGetDeviceProperties(&prop, i) );
        cout << "Device " << i << ":" << endl;
        cout << "  Name        : " << prop.name << endl;
        cout << "  TCC driver  : " << (prop.tccDriver ? "true" : "false") << endl;
        cout << "  UVA enabled : " << (prop.unifiedAddressing ? "true" : "false") << endl;
        cout << "  PCI Bus ID  : " << prop.pciBusID << endl;
    }

    for(int i=0; i<ngpus; i++) {
        check( cudaSetDevice(i) );
        helloWorld<<<1,1>>>(i);
        check (cudaGetLastError() );
    }
    for(int i=0; i<ngpus; i++) {
        check( cudaSetDevice(i) );
        check( cudaDeviceSynchronize() );
    }
}

