
#include <nag_gpu.h>
#include <nag_gpu_serial.h>

#include <math.h>
#include <iostream>


using namespace std;


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
void generateRandomNumbers(int n, double *d_buff)
{
    NagGpuRandComm comm;
    NagGpuError error;
    unsigned int seed[] = {1,2,3,4,5,6};
    naggpuRandInitA(NAGGPURANDGEN_MRG32K3A, 0,0,0,0,0, seed, &comm, &error);
    checkNag(error);

    naggpuRandNormalA(n, NAGGPURANDORDER_CONSISTENT, 0.0, 1.0, d_buff, NULL, 0, &comm, &error);
    checkNag(error);

    naggpuRandCleanupA(&comm, &error);
    checkNag(error);
}
void getVectorsToAdd(int n, double * vec1, double * vec2)
{
    const int ans = 100;
    NagCPURandComm comm;
    NagGpuError error;
    unsigned int seed[] = {1,2,3,4,5,6};
    nagCPURandInitA(NAGGPURANDGEN_MRG32K3A, 0,0,0,0,0, seed, &comm, &error);
    checkNag(error);

    nagCPURandNormalA(n, 50.0, 10.0, vec1, &comm, &error);
    checkNag(error);

    nagCPURandCleanupA(&comm, &error);
    checkNag(error);
    for(int i=0; i<n; i++) {
        vec1[i] = floor(vec1[i]);
        vec2[i] = ans - vec1[i];
    }
}




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





#include "tut1_monteCarlo_solution_question1.h"
//#include "tut1_monteCarlo_solution_question2.h"
//#include "tut1_monteCarlo_solution_question3.h"
//#include "tut1_monteCarlo_solution_question4.h"
//#include "tut1_monteCarlo_solution_question5.h"
//#include "tut1_monteCarlo_solution_question6.h"
//#include "tut1_monteCarlo_solution_question7.h"



int main()
{
    cout.precision(14);

    question1();
    //question2();
    //question3();
    //question4();
    //question5();
    //question6();
    //question7();
    return 0;
}

