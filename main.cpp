//  jacobiSolver.cpp
//  cse597
//
//  Created by Amatur Rahman on 19/9/18.
//  Copyright © 2018 psu. All rights reserved.
//

#include <stdio.h>
#include<cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <cmath>


using namespace std;


void storeMatrix(float **mat, int numRows, int numCols, string filename) {
	int x, y;
	ofstream out(filename);
	
	for (y = 0; y < numRows; y++) {
		for (x = 0; x < numCols; x++) {
			out << (int)mat[y][x] <<" ";
		}
	}
	out << endl;
	out.close();
}

void loadMatrix(float **mat, int numRows, int numCols, string filename) {
	int x, y;
	ifstream in(filename);
	
	for (y = 0; y < numRows; y++) {
		for (x = 0; x < numCols; x++) {
			in >> mat[y][x];
		}
	}
	
	in.close();
}

void loadMatrix(float *mat, int numRows, string filename) {
	int y;
	ifstream in(filename);
	
	for (y = 0; y < numRows; y++) {
			in >> mat[y];
	}
	
	in.close();
}


void print(float **mat, int numRows, int numCols){
	cout<<"Printing matrix..."<<endl;
	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++) {
			cout << mat[y][x] << "\t\t";
		}
		cout<<endl;
	}
	cout<<"\n"<<endl;
}

void print(float *mat, int numRows){
	cout<<"Printing matrix..."<<endl;
	for (int y = 0; y < numRows; y++) {
		cout << mat[y] << endl;
	}
	cout<<"\n"<<endl;
}

void print(int *mat, int numRows){
	cout<<"Printing matrix..."<<endl;
	for (int y = 0; y < numRows; y++) {
		cout << mat[y] << endl;
	}
	cout<<endl<<endl;
}


void matmul(float **A, float *B, float *C, int Dim){
	int i, k;
	for (i = 0; i < Dim; i++)		/* then multiply them together */
		for (k = 0; k < Dim; k++)
			C[i] += A[i][k] * B[k];
}


float getError(float **A, float* B, float *C, float *x, int Dim){
	matmul(A, x, C, Dim);
	float sum = 0.0;
	for (int i = 0; i<Dim; i++) {
		sum += (C[i] - x[i])*(C[i] - x[i]);
	}
	sum = sqrt(sum);
	return sum;
}



void swapRows(float** mat, int row1, int row2){
	float * tmp = mat[row1];
	mat[row1] = mat[row2];
	mat[row2] = tmp;
}
void swapRows(float* mat, int row1, int row2){
	float tmp = mat[row1];
	mat[row1] = mat[row2];
	mat[row2] = tmp;
}

void LUSolver(float ** A, float *B, int N, int eps, float *x){
	clock_t lu_st, lu_et, lu_t;
	
	//init L
	//change A to U
	
	lu_st = clock();
	float **L = (float **)calloc( N, sizeof(float *));
	
	for(int i = 0; i < N; i++ )
	{
		L[i] = (float *)calloc( N, sizeof(float));
		L[i][i] = 1;
	}
	
	int k = 0; /* Initialization of the pivot column */
	
	for(k = 0; k < N; k++){
		/* Find the k-th pivot: */
		int i_max = k;
		
		// find the largest element of this column, iterate through all row
		for(int i = k; i<N; i++){
			if(fabs(A[i][k]) > fabs(A[i_max][k])){
				i_max = i;
			}
		}
		if (fabs(A[i_max][k]) < eps) return; //fail
	
	
		if (fabs(A[i_max][k]) != k){
			swapRows(A, k, i_max);
			swapRows(B, k, i_max);
			
		}
		for (int i = k + 1; i<N; i++){
			float f = A[i][k] / A[k][k];
			L[i][k] = f;
			A[i][k] = f;
			
			
			for (int j = k + 1; j < N ; j++){
				A[i][j] = A[i][j] - A[k][j] * f;
			}
		}
		
		
	}
	
	lu_et = clock();
	lu_t = (double)(lu_et - lu_st);
	printf("Total ticks taken by CPU for LU Decomposition Only: %ld\n", lu_t );
	
	for (int i=0; i<N; i++) {
		for (int j=0; j<i; j++) {
			A[i][j] = L[i][j];
		}
	}
	
	
	//print(L, N, N);
	//decomposition up to this part
	
	// now solving
	
	//Ly = B
	
	
    
    
    lu_st = clock();
	
	for (int i = 0; i < N; i++) {
		x[i] = B[i];
		
		for (int k = 0; k < i; k++){
			x[i] -= L[i][k] * x[k];
		}
			
	}
	
	lu_et = clock();
	lu_t = (double)(lu_et - lu_st);
	printf("Total ticks taken by CPU for Forward Substitution: %ld\n", lu_t );
	
	
	lu_st = clock();
	

	//Ux = y
	for (int i = N - 1; i >= 0; i--) {
		for (int k = i + 1; k < N; k++){
			x[i] -= A[i][k] * x[k];
		}
		
		x[i] = x[i] / A[i][i];
	}
	
	
	lu_et = clock();
	lu_t = (double)(lu_et - lu_st);
	printf("Total ticks taken by CPU for Backward Substitution: %ld\n", lu_t );
	
	free(L);
}



void jacobiSolve ( int n, float **A, float *b, float epsilon, int maxit, int *numit, float *x )
{
    float *dx,*y;
    dx = (float*) calloc(n,sizeof(float));
    y = (float*) calloc(n,sizeof(float));
    int i,j,k;

	//random
	for(int j=0; j<n; j++)
	{
		x[j] = (float)rand()/(float)(RAND_MAX)*1.0;
     }
	//LUSolver(A, b, n, epsilon, x);
	

    // Note that we go through to our max iterations
    for(k=0; k<maxit; k++)
    {
        float totSum = 0.0;
        float localSum = 0.0;
        float localInd = 0.0;
        for(i=0; i<n; i++)
        {
            dx[i] = b[i];
            for(j=0; j<n; j++)
            {
                dx[i] -= A[i][j]*x[j]; 
            }
            dx[i] /= A[i][i];
            y[i] += dx[i];
            
            // Create a residual from part of the domain (when the indices  are 0, 5, 10, ...)
            if ( (i % 5) == 0)
            {
                localSum += ( (dx[i] >= 0.0) ? dx[i] : -dx[i]);

            }

            // Create a residual from a single point
            if (i == n/2)
            {
                localInd += ( (dx[i] >= 0.0) ? dx[i] : -dx[i]);
            }

            // Create a residual over all of the domain
            totSum += ( (dx[i] >= 0.0) ? dx[i] : -dx[i]);
        }
       
        // Update x
        for(i=0; i<n; i++) x[i] = y[i];
        
        // Print the residuals to the screen
        //printf("%4d\t%.3e\t%.3e\t%.3e \n",k,totSum,localSum,localInd);

        // Break out if we reach our desired tolerance
        if(totSum <= epsilon) break;
    }
    *numit = k+1;
    free(dx); free(y);
}




//~ void jacobiPoisson ( int n, float **A, float *b, float epsilon, int maxit, int *numit, float *x )
//~ {
    //~ float *dx,*y;
    //~ dx = (float*) calloc(n,sizeof(float));
    //~ y = (float*) calloc(n,sizeof(float));
    //~ int i,j,k;
    
    //~ float ** U = (float**) calloc(n,sizeof(float *));
	//~ for (int i = 0; i< n; i++){
		//~ U[i] =(float*) calloc(n,sizeof(float));
	//~ }
    //~ // Note that we go through to our max iterations
    //~ for(k=0; k<maxit; k++)
    //~ {
        //~ float totSum = 0.0;
        //~ float localSum = 0.0;
        //~ float localInd = 0.0;
        //~ for(i=0; i<n; i++)
        //~ {
            //~ //dx[i] = b[i];
            //~ for(j=0; j<n; j++)
            //~ {
				//~ if(i-1 >=0 && i+1 < n && j-1>=0 && j+1 < n){
				  //~ U[i][j] =  ( U[i-1][j] + U[i+1][j] + U[i][j-1] + U[i][j+1] + b[i*n + j] )/4;
			  //~ }
               //~ // dx[i] -= A[i][j]*x[j]; 
            //~ }
            //~ //dx[i] /= A[i][i];
            //~ //y[i] += dx[i];
            
            
          

        //~ }
       

        //~ // Break out if we reach our desired tolerance
        //~ //if(totSum <= epsilon) break;
    //~ }
    //~ int t = 0;
    //~ for(int i=0; i<n; i++) {
			//~ for(j=0; j<n; j++) {
				//~ x[t++] = U[i][j]; 
			//~ }
		//~ }
    //~ //*numit = k+1;
    //~ //free(dx); free(y);
//~ }





void jacobi(float ** A, float * B, int m, int n, float *x, float eps, int maxit){
	memset(x, 0, n*sizeof(*x)); //init guess
	float* sigma = (float*) calloc(n,sizeof(float));
	
	float *C = (float *) malloc( n * sizeof(float));
	int it = 0;
	
	int k = 0;
	do{
		it++;
		for (int i=0; i<n; i++) {
			sigma[i] = B[i];
			for (int j = 0; j < n; j++) {
				//if(j!=1){
					sigma[i] -= A[i][j] * x[j];
					
				//}
			}
			sigma[i] /= A[i][i];
			
			x[i] += sigma[i];
		}
		k = k + 1;
		//print(x, n);
		//printf("%f", getError(A, B, C, x, n));
		
		if(getError(A, B, C, x, n)<=eps || it >= maxit){
			break;
		}
		
		
	}while(true);
	
	
}




int main(){
	//-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-//
	// Set up the size of the matrix to be solved
	//int n = 1;
	int N;
	//printf("Enter the rank of the matrix:\n");
	//scanf("%d",&n);
	N = 90;
	
	
	int i;
	//init matrix
	
	//-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-//
	// Set up the Ax=b and random variables
	float **A, **A2;
	float *B, *B2;
	float *X;
	srand(0);

	//-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-//
	// Allocate memory for A matrix 
	
	A = (float**) calloc(N,sizeof(float*));
	A2 = (float**) calloc(N,sizeof(float*));
    for(i =0; i <N; i++){
		A[i] = (float*) calloc(N,sizeof(float));
		A2[i] = (float*) calloc(N,sizeof(float));
		
	}
	B = (float*) calloc(N,sizeof(float));
	B2 = (float*) calloc(N,sizeof(float));
	X = (float*) calloc(N,sizeof(float));
        
    // Randomly Initialize the A matrix
	//~ int countA,countB;
	//~ for (countA=0; countA<N; countA++) 
	//~ {
		//~ for (countB=0; countB<N; countB++) 
		//~ {
			//~ A[countA][countB]=(float)rand()/(float)(RAND_MAX)*1.0;
	
		//~ }
	//~ }
	   
	//-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-//
	// fill up A matrix - Poisson finite diff
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			if (i == j){
				A[i][j] = 4;
			}else if (abs(i-j) == 1){
				A[i][j] = -1;
			}
		}
	}
	


//~ for (int i = 0; i < N; i++) {
		//~ for (int j = 0; j < N; j++) {
    //~ if (i == j){
        //~ A[i][j] = 4;
    //~ }else if (i == j+1){
        //~ A[i][j] = -1;
	//~ }else if (i == j-1){
        //~ A[i][j]= -1;
	//~ }
    //~ else if (i == j+n){
        //~ A[i][j] = -1;
	//~ }
    //~ else if (i == j-n){
        //~ A[i][j] = -1;
	//~ }
   //~ }
//~ }

//~ for (int i = n; i < N-1; i=i+n) {
		//~ for (int j = n; j < N-1; j=j+n) {
        //~ A[i+1][j] = 0;
        //~ A[i][j+1] = 0;
	//~ }
//~ }
	//print(A, N, N);
	
	
	//-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-//
	// fill up B matrix 
	//loadMatrix(B, N, "mat.txt");
	
	for(int i =0; i<N; i++)
    {
        B[i] = (float)rand()/(float)(RAND_MAX)*1.0;
        B2[i] = B[i];
    }
	
	
	
	// fill up B matrix
	 //~ for(i =0; i<N; i++)
    //~ {
        //~ B[i] = 2.0*N;
    //~ }
    
    
    
	//-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-//
	// Set our tolerance and maximum iterations
    float eps = 1.0e-4;
    int maxit = 2*N*N;
    
    //-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-//
	//print(A, N, N);
	//print(B, N);
	//-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-//
	
	
	
	clock_t jacobi_start_t, jacobi_end_t, jacobi_t, lu_st, lu_et, lu_t;
    
    
    lu_st = clock();
	
	LUSolver(A, B, N, eps, X);

	lu_et = clock();
	
	lu_t = (double)(lu_et - lu_st);
	
	printf("Total ticks taken by CPU for LU Decomposition: %ld\n", lu_t );
	
	//print(X, N);

    
    jacobi_start_t = clock();

	int cnt;
    jacobiSolve(N, A, B, eps, maxit, &cnt, X);
	//jacobi(A, B, N, N, X, eps, maxit);
	jacobi_end_t = clock();
	jacobi_t = (double)(jacobi_end_t - jacobi_start_t);
	
	printf("Total ticks taken by CPU for Jacobi: %ld\n", jacobi_t );
	
	//print(X, N);
	//-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-//
	
	
	return 0;
}
