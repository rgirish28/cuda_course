
#include <nag_gpu.h>
#include <nag_gpu_serial.h>
#include <math.h>
#include <iostream>


using namespace std;


void checkNag(NagGpuError &error) { if(error.code != 0) { char * buff
= new char[error.msgLength]; naggpuErrorCopyMsg(buff, &error); cout <<
buff << endl; delete[] buff; exit(-1); } } void
generateRandomNumbers(int n, double *d_buff) { NagGpuRandComm comm;
NagGpuError error; unsigned int seed[] = {1,2,3,4,5,6};
naggpuRandInitA(NAGGPURANDGEN_MRG32K3A, 0,0,0,0,0, seed, &comm,
&error); checkNag(error);

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

__global__ void helloworld3(){
	
	int threadIDy = threadIdx.y;
	int threadIDx = threadIdx.x;

	int blockIDx = blockIdx.x;

	int blockIDy = blockIdx.y;


	int threadIndex = blockIDy*blockDim.x*blockDim.x*blockDim.y+blockIDx*blockDim.x*blockDim.y+threadIDy*blockDim.x+threadIDx;
	
	printf("Hello from thread: Kernel Thread ID: %d, Block Thread ID: (%d,%d), Block ID: (%d,%d) \n", threadIndex, threadIDx,threadIDy, blockIDx,blockIDy);

}
__global__ void helloworld2(){
	
	int threadIDy = threadIdx.y;
	int threadIDx = threadIdx.x;
	int blockID = blockIdx.x;

	int threadIndex = blockID*blockDim.x*blockDim.y+threadIDy*blockDim.x+threadIDx;
	
	printf("Hello from thread: Kernel Thread ID: %d, Block Thread ID: (%d %d), Block ID: %d\n", threadIndex, threadIDx,threadIDy, blockID);

}


__global__ void question2(int n, double *d_ans, double *d_vec1, double *d_vec2){

  int TID = blockIdx.x*blockDim.x+threadIdx.x;

  if(TID<n){
  
  d_ans[TID] = d_vec1[TID]+d_vec2[TID];
  	
  }
   

}

__global__ void question3(int n, double *d_Z, double *d_out){

 int TID = threadIdx.x;

 for (int i=TID;i<n;i+=blockDim.x){

 d_out[TID] += d_Z[i];
}


}

__global__ void question4(int n, double *d_Z, double *d_out){

 int TID = threadIdx.x;

 extern __shared__ double partials[];

 double sum =0.0;
 for (int i=TID;i<n;i+=blockDim.x){

 sum += d_Z[i];
}
 partials[TID] = sum;

 __syncthreads();

if(TID==0){

	for(int i=1;i<blockDim.x;i++){

		partials[0]+=partials[i];
	}

d_out[0] = partials[0];
}


}

__global__ void question5(int n, double *d_Z, double *d_out){

 int TID = blockIdx.x*blockDim.x+threadIdx.x;
 int total = blockDim.x*gridDim.x;

 extern __shared__ double partials[];

double sum = 0.0;
    for(int zIdx=TID; zIdx<n; zIdx += total) {
        sum += d_Z[zIdx];
    }
    partials[threadIdx.x] = sum;
    __syncthreads();

    if(threadIdx.x==0) {
        sum = 0.0;
        for(int i=0; i<blockDim.x; i++) {
            sum += partials[i];
        }

	d_out[blockIdx.x] = sum;
    }

}

__global__ void question6(double T, double r, double s0, double sigma, double K, int n, double *d_Z, double *d_out){

 int TID = blockIdx.x*blockDim.x+threadIdx.x;
 int total = blockDim.x*gridDim.x;

 extern __shared__ double partials[];

double sum = 0.0; double sqT = sqrt(T);
    for(int zIdx=TID; zIdx<n; zIdx += total) {
        double ST = s0*exp( (r-0.5*sigma*sigma)*T + sigma*sqT*d_Z[zIdx] );
        double x = (ST>K ? ST-K : 0.0);
        x = exp(-r*T)*x;
        sum += x;
     
    }
    partials[threadIdx.x] = sum;
    __syncthreads();

    if(threadIdx.x==0) {
        sum = 0.0;
        for(int i=0; i<blockDim.x; i++) {
            sum += partials[i];
        }

	d_out[blockIdx.x] = sum;
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



int main()
{
    cout.precision(14);
	 
    dim3 B;

    B.x=2;B.y=5;

  //  helloworld2<<<3,B>>>();

    dim3 G;

    G.x=2;G.y=3;
	
  //  helloworld3<<<G,B>>>();

    cudaDeviceSynchronize();

    int n = 2013;
    
    double *ans,*vec1,*vec2;
    
    ans = new double[n];
    vec1 = new double[n];
    vec2 = new double[n];

    getVectorsToAdd(n,vec1,vec2);

    double * d_ans,* d_vec1,* d_vec2;

    cudaMalloc((void**)&d_ans, sizeof(double)*n);
    cudaMalloc((void**)&d_vec1, sizeof(double)*n);
    cudaMalloc((void**)&d_vec2, sizeof(double)*n);

    cudaMemcpy(d_vec1,vec1,sizeof(double)*n,cudaMemcpyHostToDevice);

    cudaMemcpy(d_vec2,vec2,sizeof(double)*n,cudaMemcpyHostToDevice);
    
   //question2<<<13,256>>>(n,d_ans,d_vec1,d_vec2);

    cudaMemcpy(ans,d_ans,sizeof(double)*n,cudaMemcpyDeviceToHost);

   int flag =0;
   for(int i=0;i<n;i++){
      if(ans[i]!=100)
      {flag =1;}
   }



  int nthds = 672;
  int nblks = 140;

  n = 100000000;
  double r = 0.02;
  double T = 1.0;
  double sigma = 0.09;
  double K = 100.0;
  double s0 = 100.0;  

  double *d_z,*d_out;

  ans = new double[nblks];

  cudaMalloc((void**)&d_z,sizeof(double)*n);
  cudaMalloc((void**)&d_out,sizeof(double)*nblks);



  generateRandomNumbers(n,d_z);

 cudaMemset (d_out, 0, nthds*sizeof(double) );

  //question3<<<1,nthds>>>(n,d_z,d_out);
  //question4<<<1,nthds,sizeof(double)*nthds>>>(n,d_z,d_out);
  //question5<<<nblks,nthds,sizeof(double)*nthds>>>(n,d_z,d_out);
  question6<<<nblks,nthds,sizeof(double)*nthds>>>(T, r, s0, sigma, K,  n, d_z, d_out);

  cudaMemcpy(ans,d_out,sizeof(double)*nblks,cudaMemcpyDeviceToHost);


 for(int i=1;i<nblks;i++){
    ans[0]+=ans[i];
 }

 printf("%f",ans[0]/n);



}



