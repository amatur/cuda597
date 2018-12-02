#include<stdio.h>
#include<stdlib.h>

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

