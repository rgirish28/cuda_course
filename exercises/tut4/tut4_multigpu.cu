
#include <math.h>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <nag_gpu.h>
#include <omp.h>

using namespace std;


#define check(cuErr)  checkImpl( (cuErr), __FILE__, __LINE__)

void checkImpl(cudaError_t err, const char *file, int line)
{
#if defined(_DEBUG)
    cudaDeviceSynchronize();
    err = cudaGetLastError();
#endif
    if(err!=cudaSuccess) {
        cout << "CUDA error in " << file << " at line " << line << ":" << endl;
        cout << cudaGetErrorString(err) << endl;
        exit(-1);
    }
}
void checkNag(NagGpuError &error)
{
    if(error.code != 0) {
        char * buff = new char[error.msgLength];
        naggpuErrorCopyMsg(buff, &error);
        cout << buff << endl;
        delete[] buff;
        exit(-1);
    }
}
void generateRandomNumbers(int n, unsigned int skip, float *d_buff)
{
    NagGpuRandComm comm;
    NagGpuError error;
    unsigned int seed[] = {1,2,3,4,5,6};
    naggpuRandInitA(NAGGPURANDGEN_MRG32K3A, 0,0,0,0,skip, seed, &comm, &error);
    checkNag(error);

    naggpuRandNormalA_sp(n, NAGGPURANDORDER_CONSISTENT, 0.0f, 1.0f, d_buff, NULL, 0, &comm, &error);
    checkNag(error);

    naggpuRandCleanupA(&comm, &error);
    checkNag(error);
}
void getPeerAccessDevices(int &dev0, int &dev1)
{
    dev0 = -1, dev1 = -1;
    int ngpus = 0;
    check( cudaGetDeviceCount(&ngpus) );
    cout << "Machine has " << ngpus << " GPUs. Checking for peer access ..." << endl;
    if(ngpus<=1) {
        cout << "  Can't do peer access with only one GPU!" << endl;
        exit(-1);
    }

    for(int d0=0; d0<ngpus; d0++) {
        for(int d1=d0+1; d1<ngpus; d1++) {
            int can0 = 0, can1 = 0;
            check( cudaDeviceCanAccessPeer(&can0, d0, d1) );
            check( cudaDeviceCanAccessPeer(&can1, d1, d0) );
            if(can0 && can1) {
                dev0 = d0;
                dev1 = d1;
                cout << "  ... devices " << d0 << " and " << d1 << " can do peer access!" << endl << endl;
                return;
            }
        }
    }
    cout << "  No devices can do peer access! Exiting program ..." << endl;
    exit(-1);
}




int main()
{


    return 0;
}

