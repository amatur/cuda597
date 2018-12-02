#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cstring>
#include<mpi.h>
#include "matrix_util.h"
using namespace std;



void jacobiSolve(int n, double** A, double* B, double* x, double eps = 1e-10, int maxit = 100){
	memset(x, 0, n*sizeof(*x)); //init guess
	
	//random initialization
	for(int j=0; j<n; j++)
	{
		x[j] = (double)rand()/(double)(RAND_MAX)*1.0;
    }
    
	double* sigma = (double*) calloc(n,sizeof(double));
	
	double* y = (double*) calloc(n,sizeof(double));
	
	double *C = (double *) malloc( n * sizeof(double));
	int it = 0;
	
	int k = 0;
	do{
		it++;
		
		double error = 0.0;
		for (int i=0; i<n; i++) {
			sigma[i] = B[i];
			for (int j = 0; j < n; j++) {
				//if(j!=1){
					sigma[i] -= A[i][j] * x[j];
					
				//}
			}
			sigma[i] /= A[i][i];
			
			y[i] += sigma[i];
			
			
            error += ( (sigma[i] >= 0.0) ? sigma[i] : -sigma[i]);
		}
		k = k + 1;
		//print(x, n);
		
       
        // Update x
        for(int i=0; i<n; i++) x[i] = y[i];

		if(error <=eps || it >= maxit){
			break;
		}
		
		
	}while(true);
	
	
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


void sparse(double **sparseMatrix, int dim, double **compactMatrix){

//~ Two vectors, i and j, that specify the row and column subscripts
//~ One vector, s, containing the real or complex data you want to store in the sparse matrix. Vectors i, j and s should all have the same length.
//~ Two scalar arrays, m and n, that specify the dimensions of the sparse matrix to be created
//~ An optional scalar array that specifies the maximum amount of storage that can be allocated for this sparse array

    int size = 0; 
    for (int i = 0; i < dim; i++) 
        for (int j = 0; j < dim; j++) 
            if (sparseMatrix[i][j] != 0) 
                size++;
    compactMatrix = (double**) calloc(3 ,sizeof(double*));
    for(int i = 0; i < 3; i++){
		compactMatrix[i] = (double*) calloc(size,sizeof(double));
		
	}
    
    
    // Making of new matrix 
    int k = 0; 
    for (int i = 0; i < dim; i++) 
        for (int j = 0; j < dim; j++) 
            if (sparseMatrix[i][j] != 0) 
            { 
                compactMatrix[0][k] = i; 
                compactMatrix[1][k] = j; 
                compactMatrix[2][k] = sparseMatrix[i][j]; 
                k++; 
            } 
  
    for (int i=0; i<3; i++) 
    { 
        for (int j=0; j<size; j++) 
            printf("%lf ", compactMatrix[i][j]); 
  
        printf("\n"); 
    } 
}         
               
                


int main(int argc, char* argv[]){
    double t_start, t_end, time_secs;
    t_start = MPI_Wtime();

     int n = strtol(argv[1], NULL, 10);
    
    int N = n*n;
    double **A;
    double *b, *x;
    
	srand(0);
	
	
    //clock_t start, end;
    //start = clock();		//time count starts 
    
    //-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-//
	// Allocate memory for A matrix 
	 //-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-//
	//~ A = (double**) calloc(N,sizeof(double*));
    //~ A2 = (double**) calloc(N,sizeof(double*));
    //~ for(int i =0; i <N; i++){
		//~ A[i] = (double*) calloc(N,sizeof(double));
		//~ A2[i] = (double*) calloc(N,sizeof(double));
		//~ 
	//~ }
	init2d(&A, N);
    fillA_poisson(A, n);
    
    init1d(&b, N);
    init1d(&x, N);
    
    //fill b
    for(int i =0; i<N; i++)
    {
        b[i] = (double)rand()/(double)((RAND_MAX)*1.0);
    }
	
    
    
    //printf("A Matrix: \n");
    //print(A, N);
    
    //printf("b Matrix: ");
    //print(b, N);


    
    //printf("x Matrix: ");
    //print(x, N);
    
    //-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-//
	// Set our tolerance and maximum iterations
    double eps = 1.0e-4;
    int maxit = 2*N*N;
    
    //-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-//
	// Run jacobi
	jacobiSolve(N, A, b, x, eps, maxit);
	//printf("x Matrix: ");
    //print(x, N);
    
    //end = clock();//time count stops 
    
    //double total_time = ((double) (end - start))/ CLOCKS_PER_SEC ;//calulate total time
    
    //printf("nTime taken: %.20lf seconds.\n", total_time); //in seconds
    t_end = MPI_Wtime();
    time_secs = t_end - t_start;
    //cout<<"Dimension \t Time(sec) <<endl; 
    //cout<< N << "\t"<< time_secs << endl;
    printf("%d %lf\n", N, time_secs); 
    
    return 0;
}


