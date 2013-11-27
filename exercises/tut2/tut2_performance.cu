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

/************************************************************************************************************************/

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


/************************************************************************************************************************/



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

     __shared__ double dp_partials[1024];

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

__device__ float d_blkSum = 0.0f;

__device__ float d_thdSum = 0.0f;

__global__ void dpSingleMonteCarloKernel(
    float S0,
    float r,
    float T,
    float sigma,
    float K,
    int n,
    float *d_Z,
    float *d_out)
{
    int thdsPerBlk = blockDim.x;	// Number of threads per block
    const int totalThds = blockDim.x*gridDim.x;
    int tidxKernel = blockIdx.x*blockDim.x + threadIdx.x;

	      	
    float sum = 0.0f;
    float sqT = sqrt(T);
    for(int zIdx=tidxKernel ; zIdx < n; zIdx += totalThds) {
        float ST = S0*exp( (r-0.5f*sigma*sigma)*T + sigma*sqT*d_Z[zIdx] );
        float x = (ST>K ? ST-K : 0.0f);
        x = exp(-r*T)*x;
        atomicAdd(&d_thdSum,x);
    }

	

}

__global__ void memCopyKernel(
    int n,
    float * d_dest,
    float * d_src)
{
   	
   const int PREF = 4;

   float prefetch[PREF];
   int rIdx,wIdx;
   rIdx = threadIdx.x;  
   wIdx = rIdx;	

   for(int i =0;i<PREF;i++){
	
	prefetch[i] = d_src[rIdx];   
	rIdx += blockDim.x;    
}


   for(int i=threadIdx.x+4*blockDim.x;i<n;i+=4*blockDim.x){

   float x = prefetch[0];
   prefetch[0] = d_src[rIdx];
   rIdx += blockDim.x;

   d_dest[wIdx] = x;
   wIdx+= blockDim.x; 


   x = prefetch[1];
   prefetch[1] = d_src[rIdx];
   rIdx += blockDim.x;

   d_dest[wIdx] = x;
   wIdx+= blockDim.x; 
   x = prefetch[2];
   prefetch[2] = d_src[rIdx];
   rIdx += blockDim.x;

   d_dest[wIdx] = x;
   wIdx+= blockDim.x; 
   x = prefetch[3];
   prefetch[3] = d_src[rIdx];
   rIdx += blockDim.x;

   d_dest[wIdx] = x;
   wIdx+= blockDim.x; 

   }	
    

   int diff = n-wIdx;
   int ctr = diff/blockDim.x;   

   for(int i=0;i<ctr;i++){

   d_dest[wIdx] = prefetch[i];
   wIdx+= blockDim.x; 
   }


}

__global__ void kernelQuestion8(int n, float * d_dest, float * d_src)
{
    const int PREF = 10;
    float pref[PREF];

    int ridx = threadIdx.x;
    for(int i=0; i<PREF; i++) {
        pref[i] = d_src[ridx];
        ridx += blockDim.x;
    }
    int widx = threadIdx.x;
    while(widx<n) {
#pragma unroll 10
        for(int i=0; i<PREF; i++) {
            float x = pref[i];
            pref[i] = d_src[ridx];
            ridx += blockDim.x;
            d_dest[widx] = x;
            widx += blockDim.x;
        }
    }
}



__global__ void dpFixedSingleMonteCarloKernel(
    float S0,
    float r,
    float T,
    float sigma,
    float K,
    int n,
    float *d_Z,
    float *d_out,
    int nloops )
{
    int thdsPerBlk = blockDim.x;	// Number of threads per block
    const int totalThds = blockDim.x*gridDim.x;
    int tidxKernel = blockIdx.x*blockDim.x + threadIdx.x;

    int offset = tidxKernel*nloops;
    extern __shared__ float dp_partials[];

    float sum = 0.0f;
    float sqT = sqrt(T);
  
    int final = offset+nloops;

    if (final>n)
    final = n;

    for(int zIdx=offset ; zIdx<final; zIdx += 1) {
        float ST = S0*exp( (r-0.5f*sigma*sigma)*T + sigma*sqT*d_Z[zIdx] );
        float x = (ST>K ? ST-K : 0.0f);
        x = exp(-r*T)*x;
        sum += x;
    }

    dp_partials[threadIdx.x] = sum;
    __syncthreads();

    if(threadIdx.x == 0) {
        sum = 0.0f;
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
	
	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
    check( cudaMalloc((void**)&d_Z, sizeof(double)*n) );
    check( cudaMalloc((void**)&d_out, sizeof(double)*nblks) );
	

    generateRandomNumbers(n, d_Z);
	cudaEventRecord(start,0);
    dpMonteCarloKernel<<<nblks, nthds, sizeof(double)*nthds>>>(S0, r, T, sigma, K, n, d_Z, d_out);
	cudaEventRecord(stop,0);
    check( cudaGetLastError() );

    double *out = new double[nblks];
    check( cudaMemcpy(out, d_out, sizeof(double)*nblks, cudaMemcpyDeviceToHost) );
	
	float time = 0.0f;
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time,start,stop);
    double sum = 0.0;
    for(int i=0; i<nblks; i++) {
        sum += out[i];
    }

    cout << "\taverage=" << sum/n << endl;
    cout << "\tTime=" << time << endl;
    check( cudaFree(d_Z) );
    check( cudaFree(d_out) );
    delete[] out;
}


void callSinglePrecisionKernel(int n)
{
    int nthds = 192;
    int nblks = 224;

    float
    r = 0.02,
    T = 1.0,
    sigma = 0.09,
    K = 100.0,
    S0 = 100.0;
  	
    float n1 = (float)(n)/(nthds*nblks);	

    int nloops =  ceil(n1);
    

    float *d_Z = NULL;
    float *d_out = NULL;
	
	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
    check( cudaMalloc((void**)&d_Z, sizeof(float)*n) );
    check( cudaMalloc((void**)&d_out, sizeof(float)*nblks) );
	

    generateRandomNumbers(n, d_Z);
	cudaEventRecord(start,0);
    dpSingleMonteCarloKernel<<<nblks, nthds, sizeof(float)*nthds>>>(S0, r, T, sigma, K, n, d_Z, d_out);
	cudaEventRecord(stop,0);
    check( cudaGetLastError() );

    float *out = new float;
    check( cudaMemcpyFromSymbol(out, d_thdSum 
, sizeof(float), 0, cudaMemcpyDeviceToHost) );
	
	float time = 0.0f;
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time,start,stop);

    cout << "\taverage=" << out[0]/n << endl;
    cout << "\tTime=" << time << endl;
    check( cudaFree(d_Z) );
    check( cudaFree(d_out) );
    delete[] out;
}


// Put your code here








void launchMemcpyKernel(int nthds, int n, float * d_dest, float * d_src)
{
    // Launch your kernel for Questions 5 and 8 here
    

    kernelQuestion8<<<1,nthds>>>(n,d_dest,d_src);

}
void launchMemcpyKernel(int nthds, int n, float2 * d_dest, float2 * d_src)
{
    // Launch your kernel for Question 6 here

    //memCopyKernel<<<1,nthds>>>(n,d_dest,d_src);
}
void launchMemcpyKernel(int nthds, int n, float4 * d_dest, float4 * d_src)
{
    // Launch your kernel for Question 7

    //memCopyKernel<<<1,nthds>>>(n,d_dest,d_src);
}



int main()
{
    cout.precision(8);
	//callDoublePrecisionKernel(100000000);
        //callSinglePrecisionKernel(100000000); 

	int n = 67108864;
 	float *d_dest,*d_src;

	cudaMalloc((void**)&d_dest,sizeof(float)*(n+1024*16));
	cudaMalloc((void**)&d_src,sizeof(float)*(n+1024*16));

	cudaMemset(d_src,1,sizeof(float)*n);

	autoTuneMemcpyKernel(n, d_dest,d_src);
    return 0;
}

