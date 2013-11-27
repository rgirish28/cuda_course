#include <nag_gpu.h>

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
void generateRandomNumbers(int n, float *d_buff)
{
    NagGpuRandComm comm;
    NagGpuError error;
    unsigned int seed[] = {1,2,3,4,5,6};
    naggpuRandInitA(NAGGPURANDGEN_MRG32K3A, 0,0,0,0,0, seed, &comm, &error);
    checkNag(error);

    naggpuRandNormalA_sp(n, NAGGPURANDORDER_CONSISTENT, 0.0f, 1.0f, d_buff, NULL, 0, &comm, &error);
    checkNag(error);

    naggpuRandCleanupA(&comm, &error);
    checkNag(error);
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

// Function prototypes
void launchMemcpyKernel(int nthds, int n, float * d_dest, float * d_src);
void launchMemcpyKernel(int nthds, int n, float2 * d_dest, float2 * d_src);
void launchMemcpyKernel(int nthds, int n, float4 * d_dest, float4 * d_src);
/**
 * Autotunes the memcpy kernel
 */
template<typename FP>
void autoTuneMemcpyKernel(int n, FP * d_dest, FP * d_src)
{
    const int NTHDS_MIN = 32;
    const int NTHDS_MAX = 1024;
    cudaEvent_t start, stop;

    check( cudaEventCreate(&start) );
    check( cudaEventCreate(&stop) );

    float bestTime = 100000.0f;
    int bestNtds=-1;

    cout << "Start autotuning from nthds=" << NTHDS_MIN << " to " << NTHDS_MAX << " ..." << endl;
    for(int nthds=NTHDS_MIN; nthds<=NTHDS_MAX; nthds+=32) {
        cudaEventRecord(start);
        launchMemcpyKernel(nthds, n, d_dest, d_src);
        cudaEventRecord(stop);
        // Wait for kernel to finish
        cudaEventSynchronize(stop);

        cudaError_t cuerr = cudaGetLastError();
        if(cuerr == cudaErrorLaunchOutOfResources) {
            // Were there any "safe" errors (e.g. running out of registers)?
            continue;
        } else if(cuerr != cudaSuccess) {
            // Were there other errors?
            cout << "\t CUDA error launching with nthds=" << nthds << ": " << cudaGetErrorString(cuerr) << endl;
            exit(-1);
        }

        float time;
        cudaEventElapsedTime(&time, start, stop);
        if(time < bestTime) {
            bestTime = time;
            bestNtds = nthds;
            cout << "\tFound new best time: nthds=" << nthds << ", time=" << time << "ms" << endl;
        }
    }
    bestTime /= 1000.0f;		// Convert to seconds
    float GB = 1073741824.0f;
    cout << "... done! Best thds is " << bestNtds << " with overall bandwidth of " << 2*n*sizeof(FP)/GB/bestTime << "GB/s" << endl;

    check( cudaEventDestroy(start) );
    check( cudaEventDestroy(stop) );
}


__global__ void dpMonteCarloKernel(
    double S0,
    double r,
    double T,
    double sigma,
    double K,
    int n,
    double *d_Z,
    double *d_out)
{
    int thdsPerBlk = blockDim.x;	// Number of threads per block
    const int totalThds = blockDim.x*gridDim.x;
    int tidxKernel = blockIdx.x*blockDim.x + threadIdx.x;

    extern __shared__ double dp_partials[];

    double sum = 0.0;
    double sqT = sqrt(T);
    for(int zIdx=tidxKernel ; zIdx < n; zIdx += totalThds) {
        double ST = S0*exp( (r-0.5*sigma*sigma)*T + sigma*sqT*d_Z[zIdx] );
        double x = (ST>K ? ST-K : 0.0);
        x = exp(-r*T)*x;
        sum += x;
    }

    dp_partials[threadIdx.x] = sum;
    __syncthreads();

    if(threadIdx.x == 0) {
        sum = 0.0;
        for(int i=0; i<thdsPerBlk; i++) {
            sum += dp_partials[i];
        }
        d_out[blockIdx.x] = sum;
    }
}

void callDoublePrecisionKernel(int n)
{
    int nthds = 672;
    int nblks = 140;

    double
    r = 0.02,
    T = 1.0,
    sigma = 0.09,
    K = 100.0,
    S0 = 100.0;

    double *d_Z = NULL;
    double *d_out = NULL;

    check( cudaMalloc((void**)&d_Z, sizeof(double)*n) );
    check( cudaMalloc((void**)&d_out, sizeof(double)*nblks) );

    generateRandomNumbers(n, d_Z);

    cudaEvent_t start, stop;
    check( cudaEventCreate(&start) );
    check( cudaEventCreate(&stop) );

    check( cudaEventRecord(start,0) );
    dpMonteCarloKernel<<<nblks, nthds, sizeof(double)*nthds>>>(S0, r, T, sigma, K, n, d_Z, d_out);
    check( cudaGetLastError() );
    check( cudaEventRecord(stop, 0) );

    double *out = new double[nblks];
    check( cudaMemcpy(out, d_out, sizeof(double)*nblks, cudaMemcpyDeviceToHost) );


    double sum = 0.0;
    for(int i=0; i<nblks; i++) {
        sum += out[i];
    }
    float time;
    check( cudaEventElapsedTime(&time, start, stop) );
    cout << "\taverage=" << sum/n << endl;
    cout << "\tkernel runtime=" << time << "ms" << endl;

    check( cudaFree(d_Z) );
    check( cudaFree(d_out) );
    delete[] out;
    check( cudaEventDestroy(start) );
    check( cudaEventDestroy(stop) );
}



#include "tut2_performance_solution_question1.h"
//#include "tut2_performance_solution_question2.h"
//#include "tut2_performance_solution_question3.h"
//#include "tut2_performance_solution_question4.h"
//#include "tut2_performance_solution_question5.h"
//#include "tut2_performance_solution_question6.h"
//#include "tut2_performance_solution_question7.h"
//#include "tut2_performance_solution_question8.h"




void launchMemcpyKernel(int nthds, int n, float * d_dest, float * d_src)
{
    //kernelQuestion5<<<1, nthds>>>(n, d_dest, d_src);
    //kernelQuestion8<<<1, nthds>>>(n, d_dest, d_src);
}
void launchMemcpyKernel(int nthds, int n, float2 * d_dest, float2 * d_src)
{
    //kernelQuestion6<<<1, nthds>>>(n, d_dest, d_src);
}
void launchMemcpyKernel(int nthds, int n, float4 * d_dest, float4 * d_src)
{
    //kernelQuestion7<<<1, nthds>>>(n, d_dest, d_src);
}





int main()
{
    cout.precision(8);


    question1();
    //question2();
    //question3();
    //question4();
    //question5();
    //question6();
    //question7();
    //question8();


    return 0;
}

