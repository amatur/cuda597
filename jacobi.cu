#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cstring>


#include <cuda.h>
#include "matrix_util.h"
using namespace std;

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
__global__ void jacobiOnDevice(double* A, double* b, double* X_New, double* X_Old, int N){
	//int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//int my_rank = blockIdx.x;
	//int num_rows_block = blockDim.x;
	int my_rank = blockIdx.x;
	int num_rows_block =  blockDim.x ;
	//now do it for multiple passes

	for(int irow = my_rank*num_rows_block; irow<(my_rank+1)*num_rows_block; irow++  ){
		int index = irow * N;
		for(int icol=0;  icol<N; icol++){
			X_New[irow] -= X_Old[icol] * A[index + icol];
		}
		X_New[irow] = X_New[irow] / A[index + irow];
		//sigma[i] /= A[i][i];
	}
}


int main(int argc, char* argv[]){

	int numBlocks = 8;
	// initialize timing variables
	double t_start, t_end, time_secs;

	double **A, *A_1d, *b;
	double *X_New, *X_Old, *x;

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
	print(x);

	t_start = clock();

	/* ...Convert Matrix_A into 1-D array Input_A ......*/
	A_1d  = (double *)malloc(N*N*sizeof(double));
	convertTo1D(A, A_1d, N);

	int num_rows_block = N/numBlocks;

	//dim3 threadsPerBlock(16);
	// dim3 numBlocks(N / threadsPerBlock.x);

	//do sweeps until diff under tolerance
	int Iteration = 0;
	// on HOST
	//initialize auxiliary data structures
	X_New  = (double *) malloc (N * sizeof(double));
	X_Old  = (double *) malloc (N * sizeof(double));

	do{
		//#error Add GPU kernel calls here (see CPU version above)

		//jacobi<<16,1>>
		cudaDeviceSynchronize();
		Iteration += 1;
	}while( (Iteration < maxit) && (getError(X_Old, X_New, N) >= eps));
	//free(X_old);
	//free(Bloc_X);
	t_end = clock();
	time_secs = t_end - t_start;
	//cout<< "Time(sec): "<< time_secs << endl;
	print(X_New);
	cout<< time_secs << endl;
	return 0;
}
