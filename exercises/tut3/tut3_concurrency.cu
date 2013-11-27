
#include <math.h>
#include <iostream>
#include <string.h>

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


/*****************************************************************************************************/

void makeAMatrix(double *A, int nRows, int nCols)
{
    for(int c=0; c<nCols; c++) {
	    for(int r=0; r<nRows; r++) {
		    A[c*nRows+r] = r - c*c + r*c;
		}
	}
}

void makeBMatrices(double *Bs, int nRows, int nCols, int numBs)
{
    for(int m=0; m<numBs; m++) {
    for(int c=0; c<nCols; c++) {
        for(int r=0; r<nRows; r++) {
			Bs[m*nRows*nCols + c*nRows+r] = r*c - r + m*r*(c-r);
		}
	}
}
}
extern "C"
void dgemm_(const char*opA,const char*opB,int *m,int *n,int *k, double * alpha, double *A, int *lda, double * B, int * ldb, double * beta, double * C, int *ldc, int *info,int len1, int len2);

void checkCs(int m, int n, int k, int nMatrices, double * yourCs) {
	cout << "Checking all C matrices ...";
	double *A = new double[m*k];
	double *Bs = new double[k*n*nMatrices];
    double * refC= new double[m*n];
    double alpha = 1.0, beta = 0.0;
    int info = 0;

	makeAMatrix(A, m, k);
    makeBMatrices(Bs, k, n, nMatrices);
    
	for(int i=0; i<nMatrices; i++) {
       dgemm_("N","N",&m,&n,&k,&alpha,A,&m,&Bs[i*k*n],&k,&beta,refC,&m,&info,1,1);
       if(info !=0) cout << info << endl;

		for(int c=0; c<n; c++) {
		    for(int r=0; r<m; r++) {
				double ref = refC[c*m+r];
				double yr = yourCs[i*m*n + c*m+r];
				if(yr != ref) {
					cout << endl << "\tError checking C_" << i << "(" << r << "," << c << "): your val = " << yr << ", ref val = " << ref << endl;
					exit(-1);
				}
			}
		}
	}
	cout << " done!" << endl;
    delete[] A;
    delete[] Bs;
    delete[] refC;
}

	

/*****************************************************************************************************/

/*
    -- MAGMA (version 1.3.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2012

       @precisions normal d

*/

__global__ void 
fermiDgemm_v2_kernel_NN(double *C, const double *A, const double *B,  
                        int m, int n, int k, int lda, int ldb,  int ldc)
{
        const  int tx = threadIdx.x;
        const  int ty = threadIdx.y;

        const int iby = blockIdx.y * 64;
        const int ibx = blockIdx.x * 64;
        const int idt = ty * 64 + tx;

        const int tx2 = idt%16;
        const int ty2 = idt/16;

        __shared__ double Abs[64][17];
        __shared__ double  Bb[16][65];

        int tll = ty2;
        double xxA[4];
        double xxB[4];

        A += (ibx +__mul24( ty2, lda) + tx2);
        B += (tx2+ __mul24(iby + ty2 * 4, ldb ));

        #pragma unroll
        for(int y=0; y<4; y++)
                Abs[tx2+ y*16][ty2] = A[y*16] ;

        #pragma unroll
        for(int y=0; y<4; y++)
                Bb[tx2][ty2*4+y] = B[y * ldb] ;

        __syncthreads();

        double Axs[4];
        double Bxp[4];

        double Cb[16] = {0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0};

        int k1;
        for(k1=0; k1<(k-16); k1+=16)
        {
                tll+=16;
                A += lda *16  ;
                B += 16;

                #pragma unroll
                for( int y=0; y<4; y++)
                        xxA[y] = A[y*16];

                #pragma unroll
                for( int y=0; y<4; y++)
                        xxB[y] = B[y*ldb];

                #pragma unroll 
                for( int j1=0;j1<16;j1++)
                {
                        #pragma unroll
                        for( int y=0; y<4; y++)
                                Axs[y] =  Abs[tx2+y*16][j1] ;

                        #pragma unroll
                        for( int y=0; y<4; y++)
                                Bxp[y]= Bb[j1][ty2+y*16];


                        #pragma unroll 
                        for( int x=0; x<4; x++)
                        {
                                #pragma unroll 
                                for( int y=0; y<4; y++)
                                {
                                        Cb[x*4+y]  += Axs[x]*Bxp[y];
                                }
                        }
                }

                __syncthreads();
                
                #pragma unroll
                for(int y=0; y<4; y++)
                        Abs[tx2+y*16][ty2] = xxA[y]; 

                #pragma unroll
                for(int y=0; y<4; y++)
                        Bb[tx2][ty2*4 + y] = xxB[y];

                __syncthreads();
        }

        C += tx2 + ibx  + __mul24 (ty2 +  iby ,ldc);
        #pragma unroll 
        for(int j1=0;j1<16;j1++)
        {

                #pragma unroll
                for( int y=0; y<4; y++)
                        Axs[y] =  Abs[tx2 + y*16][j1] ;

                #pragma unroll
                for( int y=0; y<4; y++)
                        Bxp[y]= Bb[j1][ty2 + y*16];

                #pragma unroll 
                for( int x=0; x<4; x++)
                {
                        #pragma unroll 
                        for( int y=0;y<4; y++)
                        {
                                Cb[x*4 + y]  += Axs[x]*Bxp[y];
                        }
                }
        }
        int gy = iby + ty2;
        #pragma unroll
        for( int y=0;y<4;y++, gy+=16)
        {
                int gx = ibx + tx2; 
                #pragma unroll
                for(int x=0;x<4;x++, gx+=16)
                {
                        if (gx < m && gy < n)
                                C[x*16] = Cb[y+x*4];
                }
                C += ldc*16;
        }
}
inline void launchKernel(int m, int n, int k, const double *d_A, const double * d_B, double *d_C, cudaStream_t stream)
{
        dim3 threads( 64, 4 );
        dim3 grid(m/(64)+(m%(64)!=0),n/(64)+(n%(64)!=0));
        fermiDgemm_v2_kernel_NN<<< grid, threads, 0, stream>>>(d_C, d_A, d_B, m, n, k, m, k, m);
}


/*****************************************************************************************************/


void timeH2DCopyForA(double * d_A, double * A, int nElements)
{
	cudaEvent_t start, stop;
	check( cudaEventCreate(&start) );
	check( cudaEventCreate(&stop) );
	check( cudaEventRecord(start, 0) );
	
	for(int i=0; i<5; i++)
		check( cudaMemcpy(d_A, A, sizeof(double)*nElements, cudaMemcpyHostToDevice) );
	
	check( cudaEventRecord(stop, 0) );
	check( cudaEventSynchronize(stop) );
	float time;
	check( cudaEventElapsedTime(&time, start, stop) );
	time = time/5.0f;
	float szInGigs = (float)(sizeof(double)*nElements)/1073741824.0f;
	cout << "Copy A host->device takes " << time << "ms at " << szInGigs*1000.0f/time << "GB/s" << endl;

	check( cudaEventDestroy(start) );
	check( cudaEventDestroy(stop) );
}

void timeH2DCopyForB(double * d_B, double * B, int nElements)
{
	cudaEvent_t start, stop;
	check( cudaEventCreate(&start) );
	check( cudaEventCreate(&stop) );
	check( cudaEventRecord(start, 0) );
	
	for(int i=0; i<5; i++)
		check( cudaMemcpy(d_B, B, sizeof(double)*nElements, cudaMemcpyHostToDevice) );
	
	check( cudaEventRecord(stop, 0) );
	check( cudaEventSynchronize(stop) );
	float time;
	check( cudaEventElapsedTime(&time, start, stop) );
	time = time/5.0f;
	float szInGigs = (float)(sizeof(double)*nElements)/1073741824.0f;
	cout << "Copy B host->device takes " << time << "ms at " << szInGigs*1000.0f/time << "GB/s" << endl;

	check( cudaEventDestroy(start) );
	check( cudaEventDestroy(stop) );
}

void timeD2HCopyForC(double * C, double * d_C, int nElements)
{
	cudaEvent_t start, stop;
	check( cudaEventCreate(&start) );
	check( cudaEventCreate(&stop) );
	check( cudaEventRecord(start, 0) );
	
	for(int i=0; i<5; i++)
		check( cudaMemcpy(C, d_C, sizeof(double)*nElements, cudaMemcpyDeviceToHost) );
	
	check( cudaEventRecord(stop, 0) );
	check( cudaEventSynchronize(stop) );
	float time;
	check( cudaEventElapsedTime(&time, start, stop) );
	time = time/5.0f;
	float szInGigs = (float)(sizeof(double)*nElements)/1073741824.0f;
	cout << "Copy C device->host takes " << time << "ms at " << szInGigs*1000.0f/time << "GB/s" << endl;

	check( cudaEventDestroy(start) );
	check( cudaEventDestroy(stop) );
}
void timeH2DD2HCopyForBC(double * B, double * d_B, int nBElements, double * C, double * d_C, int nCElements)
{
	cudaEvent_t start, stop;
	check( cudaEventCreate(&start) );
	check( cudaEventCreate(&stop) );
    cudaStream_t stream1, stream2;
    check( cudaStreamCreate(&stream1) );
    check( cudaStreamCreate(&stream2) );
	
	check( cudaEventRecord(start, 0) );
	for(int i=0; i<5; i++) {
        check( cudaMemcpyAsync(d_B, B, sizeof(double)*nBElements, cudaMemcpyHostToDevice, stream1) );
		check( cudaMemcpyAsync(C, d_C, sizeof(double)*nCElements, cudaMemcpyDeviceToHost, stream2) );
    }
	check( cudaEventRecord(stop, 0) );
	check( cudaEventSynchronize(stop) );
    float time;
	check( cudaEventElapsedTime(&time, start, stop) );
	time = time/5.0f;
	float szInGigs = (float)(sizeof(double)*(nCElements+nBElements))/1073741824.0f;
	cout << "Bidirectional Copy B->device,C->host takes " << time << "ms at " << szInGigs*1000.0f/time << "GB/s" << endl;

	check( cudaEventDestroy(start) );
	check( cudaEventDestroy(stop) );
}

void timeKernel(double * d_A, double *d_B, double * d_C, int m, int n, int k)
{
	cudaEvent_t start, stop;
	check( cudaEventCreate(&start) );
	check( cudaEventCreate(&stop) );
	// Warm-up so we get accurate timings
	launchKernel(m,n,k,d_A,d_B,d_C, 0);

	check( cudaEventRecord(start, 0) );
	for(int i=0; i<5; i++) {
		launchKernel(m,n,k,d_A,d_B,d_C, 0);
	}	
	check( cudaEventRecord(stop, 0) );
	check( cudaEventSynchronize(stop) );
	float time;
	check( cudaEventElapsedTime(&time, start, stop) );
	time = time/5.0f;
	cout << "Kernel execution takes " << time << "ms" << endl;

	check( cudaEventDestroy(start) );
	check( cudaEventDestroy(stop) );
}
#include <omp.h>
void timeH2HCopyForStageUpld(double * stageUpld, double * B, int nElements)
{
	double start = omp_get_wtime();
	for(int i=0; i<5; i++)
		memcpy(stageUpld, B, sizeof(double)*nElements);
	
	double stop = omp_get_wtime();
	double time = (stop-start)*1000;
	time = time/5.0f;
	float szInGigs = (float)(sizeof(double)*nElements)/1073741824.0f;
	cout << "Copy stageUpload host->host takes " << time << "ms at " << szInGigs*1000.0f/time << "GB/s" << endl;
}

void timeH2HCopyForStageDownld(double * C, double * stageDownld, int nElements)
{
	double start = omp_get_wtime();
	for(int i=0; i<5; i++)
		memcpy(C, stageDownld, sizeof(double)*nElements);
	
	double stop = omp_get_wtime();
	double time = (stop-start)*1000;
	time = time/5.0f;
	float szInGigs = (float)(sizeof(double)*nElements)/1073741824.0f;
	cout << "Copy stageDownld host->host takes " << time << "ms at " << szInGigs*1000.0f/time << "GB/s" << endl;
}


/*****************************************************************************************************/
void question1(int m, int n, int k, int nMatrices){

double * A = new double[m*k];
double * B = new double[k*n*nMatrices];
double * C = new double[m*n*nMatrices];

double *d_A,*d_Bi,*d_Ci;

cudaMalloc((void **)&d_A,sizeof(double)*m*k);
cudaMalloc((void **)&d_Bi,sizeof(double)*n*k);
cudaMalloc((void **)&d_Ci,sizeof(double)*m*n);


makeAMatrix(A, m, k);
makeBMatrices(B,k,n,nMatrices);


timeH2DCopyForA(d_A, A, m*k);
timeH2DCopyForB(d_Bi, B, n*k);
timeD2HCopyForC(C, d_Ci, m*n);
timeKernel(d_A, d_Bi, d_Ci, m, n, k);


cudaEvent_t event1, event2;

cudaEventCreate(&event1);
cudaEventCreate(&event2);
cudaEventRecord(event1, 0); 
cudaMemcpy(d_A, A, sizeof(double)*m*k, cudaMemcpyHostToDevice);

for(int i=0;i<nMatrices;i++){

cudaMemcpy(d_Bi, &B[i*n*k], sizeof(double)*n*k, cudaMemcpyHostToDevice);
launchKernel(m, n, k, d_A, d_Bi, d_Ci, 0);
cudaMemcpy(&C[i*m*n],d_Ci, sizeof(double)*m*n, cudaMemcpyDeviceToHost);
}
cudaEventRecord(event2, 0);

cudaEventSynchronize(event2); 

float dt_ms;
cudaEventElapsedTime(&dt_ms, event1, event2);
cout<< "The time is : "<< dt_ms<<endl;

checkCs(m, n, k, nMatrices, C);

}
void question2(int m, int n, int k, int nMatrices){

double * A = new double[m*k];
double * B;
double * C;


makeAMatrix(A, m, k);

cudaHostAlloc 	((void **) &B,sizeof(double)*k*n*nMatrices,cudaHostAllocMapped);	 	

cudaHostAlloc 	((void **) &C,sizeof(double)*m*n*nMatrices,cudaHostAllocMapped);


double *d_A,*d_B,*d_C;

cudaMalloc((void **)&d_A,sizeof(double)*m*k);

makeBMatrices(B,k,n,nMatrices);

cudaHostGetDevicePointer((void**) &d_B, B, 0);
cudaHostGetDevicePointer((void**) &d_C, C, 0);


cudaEvent_t event1, event2;

cudaEventCreate(&event1);
cudaEventCreate(&event2);


cudaEventRecord(event1, 0); 
cudaMemcpy(d_A, A, sizeof(double)*m*k, cudaMemcpyHostToDevice);

for(int i=0;i<nMatrices;i++){

launchKernel(m, n, k, d_A, &d_B[i*n*k], &d_C[i*m*n], 0);

}
cudaEventRecord(event2, 0);

cudaEventSynchronize(event2); 

float dt_ms;
cudaEventElapsedTime(&dt_ms, event1, event2);
cout<< "The time is : "<< dt_ms<<endl;

checkCs(m, n, k, nMatrices, C);

}

void question4(int m, int n, int k, int nMatrices){

double * A = new double[m*k];
double * B = new double[k*n*nMatrices];
double * C new double[m*n*nMatrices]= ;

double *B_staged;
double *C_staged;

makeAMatrix(A, m, k);

cudaHostAlloc 	((void **) &B_staged,sizeof(double)*k*n,cudaHostAllocWriteCombined);	 	

cudaHostAlloc 	((void **) &C_staged,sizeof(double)*m*n,cudaHostAllocDefault);


double *d_A,*d_B[3],*d_C[3];

cudaStream_t stream[3];

for(int j=0;j<3;j++){
	cudaStreamCreate(&stream[j]);
	cudaMalloc((void **)&d_B[j],sizeof(double)*n*k);
	cudaMalloc((void **)&d_C[j],sizeof(double)*m*n);
}

cudaMalloc((void **)&d_A,sizeof(double)*m*k);

makeBMatrices(B,k,n,nMatrices);

timeH2HCopyForStageDownld(double * C, double * stageDownld, int nElements)


timeH2DCopyForA(d_A, A, m*k);
timeH2HCopyForStageUpld(B_staged, B ,n*k);
timeH2HCopyForStageDownld(C, C_staged,n*m);
timeKernel(d_A, d_B[0], d_C[0], m, n, k);
timeH2DD2HCopyForBC(B_staged, d_B[0], n*k, C_staged, d_C[0], m*n);


cudaEvent_t event1, event2, event3;

cudaEventCreate(&event1,cudaEventDisableTiming);
cudaEventCreate(&event2,cudaEventDisableTiming);
cudaEventCreate(&event3,cudaEventDisableTiming);

cudaEventRecord(event1, 0); 
cudaMemcpy(d_A, A, sizeof(double)*m*k, cudaMemcpyHostToDevice);


for(int i=0,j=0;i<nMatrices;i=i+1){
		
		check(cudaMemcpyAsync(d_B[j],&B[i*n*k],n*k*sizeof(double),cudaMemcpyHostToDevice,stream[j])); 		
		
		launchKernel(m, n, k, d_A, d_B[j], d_C[j], stream[j]);
		check(cudaMemcpyAsync(&C[i*m*n],d_C[j],n*m*sizeof(double),cudaMemcpyDeviceToHost,stream[j]));
		j++;
		
		if(j>=3)
		j=0;

}
cudaEventRecord(event2, 0);

cudaEventSynchronize(event2); 

float dt_ms;
cudaEventElapsedTime(&dt_ms, event1, event2);
cout<< "The time is : "<< dt_ms<<endl;

checkCs(m, n, k, nMatrices, C);

}

void question3(int m, int n, int k, int nMatrices){

double * A = new double[m*k];
double * B;
double * C;


makeAMatrix(A, m, k);

cudaHostAlloc 	((void **) &B,sizeof(double)*k*n*nMatrices,cudaHostAllocDefault);	 	

cudaHostAlloc 	((void **) &C,sizeof(double)*m*n*nMatrices,cudaHostAllocDefault);


double *d_A,*d_B[3],*d_C[3];

cudaStream_t stream[3];

for(int j=0;j<3;j++){
	cudaStreamCreate(&stream[j]);
	cudaMalloc((void **)&d_B[j],sizeof(double)*n*k);
	cudaMalloc((void **)&d_C[j],sizeof(double)*m*n);
}

cudaMalloc((void **)&d_A,sizeof(double)*m*k);

makeBMatrices(B,k,n,nMatrices);

timeH2DCopyForA(d_A, A, m*k);
timeH2DCopyForB(d_B[0], B, n*k);
timeD2HCopyForC(C, d_C[0], m*n);
timeKernel(d_A, d_B[0], d_C[0], m, n, k);
timeH2DD2HCopyForBC(B, d_B[0], n*k, C, d_C[0], m*n);


cudaEvent_t event1, event2;

cudaEventCreate(&event1);
cudaEventCreate(&event2);


cudaEventRecord(event1, 0); 
cudaMemcpy(d_A, A, sizeof(double)*m*k, cudaMemcpyHostToDevice);


for(int i=0,j=0;i<nMatrices;i=i+1){
		
		check(cudaMemcpyAsync(d_B[j],&B[i*n*k],n*k*sizeof(double),cudaMemcpyHostToDevice,stream[j])); 		
		launchKernel(m, n, k, d_A, d_B[j], d_C[j], stream[j]);
		check(cudaMemcpyAsync(&C[i*m*n],d_C[j],n*m*sizeof(double),cudaMemcpyDeviceToHost,stream[j]));
		j++;
		
		if(j>=3)
		j=0;

}
cudaEventRecord(event2, 0);

cudaEventSynchronize(event2); 

float dt_ms;
cudaEventElapsedTime(&dt_ms, event1, event2);
cout<< "The time is : "<< dt_ms<<endl;

checkCs(m, n, k, nMatrices, C);

}


int main()
{
	const int m = 800, n = 2880, k = 896, nMatrices=20;
	check( cudaSetDeviceFlags(cudaDeviceMapHost) );
	
	
	// Start your code here!
        question3(m, n, k, nMatrices);


	return 0;
}

