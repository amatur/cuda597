#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cstring>
#include <cuda.h>
#include<assert.h>
#include <curand.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
//using namespace std;

__device__ int flag;

void init2d(float ***A, int n){
	float** B = (float**) calloc(n,sizeof(float*));
    for(int i =0; i <n; i++){
		B[i] = (float*) calloc(n,sizeof(float));
	}
	*A = B;
}

void init2d(float ***A, float ***A2, int n){
	float** B = (float**) calloc(n,sizeof(float*));
    for(int i =0; i <n; i++){
		B[i] = (float*) calloc(n,sizeof(float));

	}
	for (int y = 0; y < n; y++) {
		for (int x = 0; x < n; x++) {
            B[y][x] = *A2[y][x];
		}
	}
	*A = B;
}



void init1d(float **A, int n){
	float* B = (float*) calloc(n,sizeof(float));
	*A = B;
}


void print(float *mat, int numRows){

	for (int x = 0; x < numRows; x++) {
		printf("%-20.3f ", mat[x]);
	}
	printf("\n");

}

void print(float **mat, int numRows){
	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numRows; x++) {
            printf("%-20.3f ", mat[y][x]);
		}
		printf("\n");
	}
	printf("\n");
}



void convertTo1D(float** A, float* A_1d, int N){
	int k    = 0;
	for(int irow=0; irow<N; irow++)
	for(int icol=0; icol<N; icol++)
	A_1d[k++] = A[irow][icol];
}


void jacobiSolve(int n, float** A, float* B, float* x, float eps = 1e-10, int maxit = 100){
	memset(x, 0, n*sizeof(*x)); //init guess

	//random initialization
	for(int j=0; j<n; j++)
	{
		x[j] = (float)rand()/(float)(RAND_MAX)*1.0;
	}

	float* sigma = (float*) calloc(n,sizeof(float));

	float* y = (float*) calloc(n,sizeof(float));

	//float *C = (float *) malloc( n * sizeof(float));
	int it = 0;

	int k = 0;
	do{
		it++;

		float totSum = 0.0;
		float localSum = 0.0;
		float localInd = 0.0;
		for (int i=0; i<n; i++) {
			sigma[i] = B[i];
			for (int j = 0; j < n; j++) {
				//if(j!=1){
				sigma[i] -= A[i][j] * x[j];

				//}
			}
			sigma[i] /= A[i][i];

			y[i] += sigma[i];



			// Create a residual from part of the domain (when the indices  are 0, 5, 10, ...)
			if ( (i % 5) == 0)
			{
				localSum += ( (sigma[i] >= 0.0) ? sigma[i] : -sigma[i]);

			}

			// Create a residual from a single point
			if (i == n/2)
			{
				localInd += ( (sigma[i] >= 0.0) ? sigma[i] : -sigma[i]);
			}

			// Create a residual over all of the domain
			totSum += ( (sigma[i] >= 0.0) ? sigma[i] : -sigma[i]);
		}
		k = k + 1;
		//print(x, n);
		//printf("%f", getError(A, B, C, x, n));



		// Update x
		for(int i=0; i<n; i++) x[i] = y[i];

		// Print the residuals to the screen
		//printf("%4d\t%.3e\t%.3e\t%.3e \n",k,totSum,localSum,localInd);
		//getError(A, B, C, x, n)
		if(totSum <=eps || it >= maxit){
			break;
		}


	}while(true);

	free(sigma);
	free(y);
}



/**
* @brief   Randomly Initialize the A matrix
*/
void fillA_random(float **A, int n){
	int countA,countB;
	for (countA=0; countA<n; countA++)
	{
		for (countB=0; countB<n; countB++)
		{
			A[countA][countB]=(float)rand()/(float)(RAND_MAX)*1.0;
		}
	}
}

// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on GPU


void fillA_poisson(float **A, int n){

	for(int i = 0; i<n*n; i++){
		for(int j = 0; j<n*n; j++){
			if(i==j){
				A[i][j] = 4;
			}else if (i == j+1  || i == j -1 || i==j+n || i==j-n){
				A[i][j] = -1;
			}
		}
	}

	for(int i = n-1; i<n*n-1; i = i + n){
		for(int j = n-1; j<n*n-1; j = j + n){
			A[i+1][j] = 0;
			A[i][j+1] = 0;
		}
	}

}

struct saxpy_functor
{
    const float a;

    saxpy_functor(float _a) : a(_a) {}

    __host__ __device__
        float operator()(const float& x, const float& y) const {
            return -1 + 2.0* x/(float)((RAND_MAX)*1.0);
        }
};

void saxpy_fast(float A, thrust::device_vector<float>& X, thrust::device_vector<float>& Y)
{
    // Y <- A * X + Y
    thrust::transform(X.begin(), X.end(), Y.begin(), Y.begin(), saxpy_functor(A));
}


void fillB(float *b, int n){
	for(int i =0; i<n; i++)
	{
		b[i] = -1 + 2.0* (float)rand()/(float)((RAND_MAX)*1.0);
	}
}

void fillB_random_GPU(float *B, int N) {
		// Create a pseudo-random number generator
		curandGenerator_t prng;
		curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

	 // Set the seed for the random number generator using the system clock
	 curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

	 // Fill the array with random numbers on the device
	 curandGenerateUniform(prng, B, N * N);

// 	 float myrandf = curand_uniform(&(my_curandstate[idx]));
// myrandf *= (max_rand_int[idx] - min_rand_int[idx] + 0.999999);
// myrandf += min_rand_int[idx];
// int myrand = (int)truncf(myrandf);


}

float getError(float *x, float *xnew, int N)
{
	float sum = 0.0;
	for(int index=0; index<N; index++)
	sum += (xnew[index] - x[index])*(xnew[index]-x[index]);
	sum = sqrt(sum);
	return(sum);
}

// Device version of the Jacobi method
__global__ void jacobiOnDevice(float* A, float* b, float* X_New, float* X_Old, int N, float eps){

	unsigned int i, j;
	float sigma = 0, newValue;

	i = threadIdx.x + blockIdx.x * blockDim.x;
	//printf("%d  IAAAA", i);
	for (j = 0; j < N; j++) {
		if (i != j) {
			sigma = sigma + A[i*N + j] * X_Old[j];
		}
	}
	//assert(A[i*N+i] != 0);
	newValue = (b[i] - sigma) / A[i*N + i];

	if (abs(X_Old[i] - newValue) > eps) flag = 0;
	X_Old[i] = newValue;
	//newValue;

}


int main(int argc, char* argv[]){
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaError_t  cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		printf("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");

	}
	//int numBlocks = 4;
	//int blockSize = 1;
	// initialize timing variables
	float t_start, t_end, time_secs;

	float **A, *A_1d, *b;
	float *X_New, *X_Old, *x;

// gpu Copy
float *A_1d_gpu;
//float *b_gpu;
float *X_New_gpu;
float *X_Old_gpu;

	srand(0);
	int n = strtol(argv[1], NULL, 10);
	int N = n*n;
	//-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-//
	// Set our tolerance and maximum iterations
	float eps = 1.0e-4;
	int maxit = 2*N*N;


	init2d(&A, N);
	fillA_poisson(A, n);

	init1d(&b, N);

	//printf("A Matrix: \n");
	//print(A, N);

	//printf("b Matrix: ");
	//print(b, N);
	init1d(&x, N);

	//fill b
	fillB(b, n);
	jacobiSolve(N, A, b, x, eps, maxit);
	print(x, N);
	printf("Correct one\n");


		/* ...Convert Matrix_A into 1-D array Input_A ......*/
		A_1d  = (float *)malloc(N*N*sizeof(float));
		//fillA_random_GPU(A,N);
		convertTo1D(A, A_1d, N);

	// STARTING cuda
	thrust::device_vector<float> b_gpu(N * N);

	     // Fill the arrays A and B on GPU with random numbers
	  fillB_random_GPU(thrust::raw_pointer_cast(&b_gpu[0]), N);
		saxpy_fast(5, b_gpu, b_gpu);
	// on HOST
	//initialize auxiliary data structures
	X_New  = (float *) malloc (N * sizeof(float));
	X_Old  = (float *) malloc (N * sizeof(float));

	// Allocate memory on the device
	 assert(cudaSuccess == cudaMalloc((void **) &X_New_gpu, N*sizeof(float)));
	 assert(cudaSuccess == cudaMalloc((void **) &A_1d_gpu, N*N*sizeof(float)));
	 assert(cudaSuccess == cudaMalloc((void **) &X_Old_gpu, N*sizeof(float)));
	 //assert(cudaSuccess == cudaMalloc((void **) &b_gpu, N*sizeof(float)));

	 cudaError_t ct;
	 // Copy data -> device
	 ct = cudaMemcpy(X_New_gpu, X_New, sizeof(float)*N, cudaMemcpyHostToDevice);
	 assert(ct==cudaSuccess);
	 ct = cudaMemcpy(A_1d_gpu, A_1d, sizeof(float)*N*N, cudaMemcpyHostToDevice);
	 assert(ct==cudaSuccess);
	 ct = cudaMemcpy(X_Old_gpu, X_Old, sizeof(float)*N, cudaMemcpyHostToDevice);
	 assert(ct==cudaSuccess);
	 //ct = cudaMemcpy(b_gpu, b, sizeof(float)*N, cudaMemcpyHostToDevice);
	 //assert(ct==cudaSuccess);

	//  cudaStatus = cudaMemcpy(x0, dev_x0, matrixSize* sizeof(float), cudaMemcpyDeviceToHost);
	// if (cudaStatus != cudaSuccess) {
	// 	fprintf(stderr, "cudaMemcpy failed!");
	// 	goto Error;
	// }

	t_start = clock();



	int gridSize, blockSize, minGridSize;
   cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, jacobiOnDevice, 0, N);
gridSize = (N + blockSize - 1) / blockSize;
printf("min grid size %d grid size %d, block size %d",minGridSize,gridSize, blockSize);
	//dim3 threadsPerBlock(16);
	// dim3 numBlocks(N / threadsPerBlock.x);

	//do sweeps until diff under tolerance
	//gridSize = strtol(argv[2], NULL, 10);
	//blockSize = strtol(argv[3], NULL, 10);
	int Iteration = 0;
  cudaDeviceSynchronize();
	int cpuConvergenceTest = 0;
	do{
		cpuConvergenceTest = 1;
		cudaMemcpyToSymbol(flag, &cpuConvergenceTest, sizeof(int));

		//#error Add GPU kernel calls here (see CPU version above)
		jacobiOnDevice <<< 1, N >>> (A_1d_gpu, thrust::raw_pointer_cast(&b_gpu[0]), X_New_gpu, X_Old_gpu, N, eps);
		//jacobi<<16,1>>

		cudaError_t cudaStatus = cudaDeviceSynchronize();
				if (cudaStatus != cudaSuccess) {
					printf("cudaDeviceSynchronize returned error code %d after launching jacobi!\n", cudaStatus);
				}

				cudaMemcpyFromSymbol(&cpuConvergenceTest, flag, sizeof(int));

		Iteration += 1;
		//cudaMemcpy(X_New, X_New_gpu, sizeof(float)*N, cudaMemcpyDeviceToHost);
		//cudaMemcpy(X_Old, X_Old_gpu, sizeof(float)*N, cudaMemcpyDeviceToHost);

	}while( (Iteration < maxit) && !cpuConvergenceTest);
	cudaMemcpy(X_Old, X_Old_gpu, sizeof(float)*N, cudaMemcpyDeviceToHost);
	print(X_Old, N);
	// Data <- device

    // Free memory
      cudaFree(X_New_gpu); cudaFree(X_Old_gpu);
			//cudaFree(b_gpu);
			cudaFree(A_1d_gpu);
			//free(X_Old); free(A); free(A_1d);free(A); free(b);

	//free(X_old);
	//free(Bloc_X);

	cudaDeviceSynchronize();
	t_end = clock();
	time_secs = t_end - t_start;
	//cout<< "Time(sec): "<< time_secs << endl;

	printf("%lf\n", time_secs);


	// cudaEventSynchronize(stop);
	// 	float milliseconds = 0;
	// cudaEventElapsedTime(&milliseconds, start, stop);
	// 	printf("Milli %lf\n", milliseconds);
	return 0;
}
