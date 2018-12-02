#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cstring>
#include <cuda.h>
#include<assert.h>
//using namespace std;

__device__ int flag;

void init2d(double ***A, int n){
	double** B = (double**) calloc(n,sizeof(double*));
    for(int i =0; i <n; i++){
		B[i] = (double*) calloc(n,sizeof(double));
	}
	*A = B;
}

void init2d(double ***A, double ***A2, int n){
	double** B = (double**) calloc(n,sizeof(double*));
    for(int i =0; i <n; i++){
		B[i] = (double*) calloc(n,sizeof(double));

	}
	for (int y = 0; y < n; y++) {
		for (int x = 0; x < n; x++) {
            B[y][x] = *A2[y][x];
		}
	}
	*A = B;
}



void init1d(double **A, int n){
	double* B = (double*) calloc(n,sizeof(double));
	*A = B;
}


void print(double *mat, int numRows){

	for (int x = 0; x < numRows; x++) {
		printf("%-20.3f ", mat[x]);
	}
	printf("\n");

}

void print(double **mat, int numRows){
	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numRows; x++) {
            printf("%-20.3f ", mat[y][x]);
		}
		printf("\n");
	}
	printf("\n");
}



void convertTo1D(double** A, double* A_1d, int N){
	int k    = 0;
	for(int irow=0; irow<N; irow++)
	for(int icol=0; icol<N; icol++)
	A_1d[k++] = A[irow][icol];
}


void jacobiSolve(int n, double** A, double* B, double* x, double eps = 1e-10, int maxit = 100){
	memset(x, 0, n*sizeof(*x)); //init guess

	//random initialization
	for(int j=0; j<n; j++)
	{
		x[j] = (double)rand()/(double)(RAND_MAX)*1.0;
	}

	double* sigma = (double*) calloc(n,sizeof(double));

	double* y = (double*) calloc(n,sizeof(double));

	//double *C = (double *) malloc( n * sizeof(double));
	int it = 0;

	int k = 0;
	do{
		it++;

		double totSum = 0.0;
		double localSum = 0.0;
		double localInd = 0.0;
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
void fillA_random(double **A, int n){
	int countA,countB;
	for (countA=0; countA<n; countA++)
	{
		for (countB=0; countB<n; countB++)
		{
			A[countA][countB]=(double)rand()/(double)(RAND_MAX)*1.0;
		}
	}
}

void fillA_poisson(double **A, int n){

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

void fillB(double *b, int n){
	for(int i =0; i<n; i++)
	{
		b[i] = -1 + 2.0* (double)rand()/(double)((RAND_MAX)*1.0);
	}
}

double getError(double *x, double *xnew, int N)
{
	double sum = 0.0;
	for(int index=0; index<N; index++)
	sum += (xnew[index] - x[index])*(xnew[index]-x[index]);
	sum = sqrt(sum);
	return(sum);
}

// Device version of the Jacobi method
__global__ void jacobiOnDevice(double* A, double* b, double* X_New, double* X_Old, int N, double eps, int num_rows_block){

	/*
	int my_rank = blockIdx.x;
//	int num_rows_block =  blockDim.x ;
	//now do it for multiple passes

	for(int irow = my_rank*num_rows_block; irow<(my_rank+1)*num_rows_block; irow++  ){
		int index = irow * N;
		for(int icol=0;  icol<N; icol++){
			X_New[irow] -= X_Old[icol] * A[index + icol];
		}
		X_New[irow] = X_New[irow] / A[index + irow];
		//sigma[i] /= A[i][i];
	}
	memcpy(X_Old, X_New, sizeof(double)*N);
	*/

	unsigned int i, j;
	double sigma = 0, newValue;

	i = threadIdx.x + blockIdx.x * blockDim.x;

	for (j = 0; j < N; j++) {
		if (i != j) {
			sigma = sigma + A[i*N + j] * X_Old[j];
		}
	}

	newValue = (b[i] - sigma) / A[i*N + i];

	if (abs(X_Old[i] - newValue) > eps) flag = 0;
	X_Old[i] = newValue;

}


int main(int argc, char* argv[]){
	//int numBlocks = 4;
	//int blockSize = 1;
	// initialize timing variables
	double t_start, t_end, time_secs;

	double **A, *A_1d, *b;
	double *X_New, *X_Old, *x;

// gpu Copy
double *A_1d_gpu, *b_gpu;
double *X_New_gpu, *X_Old_gpu,

	srand(0);
	int n = strtol(argv[1], NULL, 10);
	int N = n*n;
	//-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-//
	// Set our tolerance and maximum iterations
	double eps = 1.0e-4;
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


		/* ...Convert Matrix_A into 1-D array Input_A ......*/
		A_1d  = (double *)malloc(N*N*sizeof(double));
		convertTo1D(A, A_1d, N);

	// STARTING cuda

	// on HOST
	//initialize auxiliary data structures
	X_New  = (double *) malloc (N * sizeof(double));
	X_Old  = (double *) malloc (N * sizeof(double));

	// Allocate memory on the device
	 assert(cudaSuccess == cudaMalloc((void **) &X_New_gpu, N*sizeof(double)));
	 assert(cudaSuccess == cudaMalloc((void **) &A_1d_gpu, N*N*sizeof(double)));
	 assert(cudaSuccess == cudaMalloc((void **) &X_Old_gpu, N*sizeof(double)));
	 assert(cudaSuccess == cudaMalloc((void **) &b_gpu, N*sizeof(double)));

	 // Copy data -> device
	 cudaMemcpy(X_New_gpu, X_New, sizeof(double)*N, cudaMemcpyHostToDevice);
	 cudaMemcpy(A_1d_gpu, A_1d, sizeof(double)*N*N, cudaMemcpyHostToDevice);
	 cudaMemcpy(X_Old_gpu, X_Old, sizeof(double)*N, cudaMemcpyHostToDevice);
	 cudaMemcpy(b_gpu, b, sizeof(double)*N, cudaMemcpyHostToDevice);

	//  cudaStatus = cudaMemcpy(x0, dev_x0, matrixSize* sizeof(double), cudaMemcpyDeviceToHost);
	// if (cudaStatus != cudaSuccess) {
	// 	fprintf(stderr, "cudaMemcpy failed!");
	// 	goto Error;
	// }

	t_start = clock();


	int num_rows_block = N/numBlocks;
	int gridSize, blockSize, minGridSize;
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, internal_jacobi_solve, 0, matrixSize);

gridSize = (N + blockSize - 1) / blockSize;
prinf("min grid size %d grid size %d, block size %d",minGridSize,gridSize, blockSize);
	//dim3 threadsPerBlock(16);
	// dim3 numBlocks(N / threadsPerBlock.x);

	//do sweeps until diff under tolerance
	int Iteration = 0;

	int cpuConvergenceTest = 0;
	do{
		cpuConvergenceTest = 1;
		cudaMemcpyToSymbol(flag, &cpuConvergenceTest, sizeof(int));

		//#error Add GPU kernel calls here (see CPU version above)
		jacobiOnDevice <<< gridSize, blockSize >>> (A_1d_gpu, b_gpu, X_New_gpu, X_Old_gpu, N, num_rows_block);
		//jacobi<<16,1>>

		cudaError_t cudaStatus = cudaDeviceSynchronize();
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching jacobi!\n", cudaStatus);
					goto Error;
				}

				cudaMemcpyFromSymbol(&cpuConvergenceTest, flag, sizeof(int));

		Iteration += 1;
		//cudaMemcpy(X_New, X_New_gpu, sizeof(double)*N, cudaMemcpyDeviceToHost);
		//cudaMemcpy(X_Old, X_Old_gpu, sizeof(double)*N, cudaMemcpyDeviceToHost);

	}while( (Iteration < maxit) && !cpuConvergenceTest);
	//cudaMemcpy(X_New, X_New_gpu, sizeof(double)*N, cudaMemcpyDeviceToHost);
print(X_New, N);
	// Data <- device


    // Free memory
    //free(X_Old); free(A); free(A_1d);free(A); free(b);
  //  cudaFree(X_New_gpu); cudaFree(X_Old_gpu); cudaFree(b_gpu); cudaFree(A_1d_gpu);

	//free(X_old);
	//free(Bloc_X);
	t_end = clock();
	time_secs = t_end - t_start;
	//cout<< "Time(sec): "<< time_secs << endl;
	print(X_New, N);

	printf("%lf\n", time_secs);
	return 0;
}
