#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<cstring>
#include<iostream>
#include "matrix_util.h"

using namespace std;

void swapRows(double** mat, int row1, int row2){
	double * tmp = mat[row1];
	mat[row1] = mat[row2];
	mat[row2] = tmp;
}
void swapRows(double* mat, int row1, int row2){
	double tmp = mat[row1];
	mat[row1] = mat[row2];
	mat[row2] = tmp;
}


void LUSolve(int dim, double** A2, double* B, double *x, double eps=1e-10){

	 /* make a copy of A2 matrix, as the algo changes A2 in place*/
	double **A = (double **)calloc( dim, sizeof(double *));

	for(int i = 0; i < dim; i++ )
	{
		A[i] = (double *)calloc( dim, sizeof(double));
	}

	for(int i = 0; i<dim; i++){
		for(int j = 0; j<dim; j++){
			A[i][j] = A2[i][j];
		}
	}


	/* Initialization of the pivot column */
	for(int k = 0; k < dim; k++){
		/* Find the k-th pivot: */
		int i_max = k;

		// find the largest element of this column, iterate through all row
		for(int i = k; i<dim; i++){
			if(fabs(A[i][k]) > fabs(A[i_max][k])){
				i_max = i;
			}
		}
		if (fabs(A[i_max][k]) < eps) return; //fail


		if (fabs(A[i_max][k]) != k){
			swapRows(A, k, i_max);
			swapRows(B, k, i_max);

		}
		for (int i = k + 1; i<dim; i++){
			double f = A[i][k] / A[k][k];
			A[i][k] = f;


			for (int j = k + 1; j < dim ; j++){
				A[i][j] = A[i][j] - A[k][j] * f;
			}
		}


	}


	for (int i = 0; i < dim; i++) {
		x[i] = B[i];

		for (int k = 0; k < i; k++){
			x[i] -= A[i][k] * x[k];
		}

	}

	//Ux = y
	for (int i = dim - 1; i >= 0; i--) {
		for (int k = i + 1; k < dim; k++){
			x[i] -= A[i][k] * x[k];
		}

		x[i] = x[i] / A[i][i];
	}

}


int main(){
	int n = 4;
	double A2[4][4] = {
						{10., -1., 2., 0.},
                        {-1., 11., -1., 3.},
                        {2., -1., 10., -1.},
                        {0.0, 3., -1., 8.}
                       };

    double b2[4] =  {6., 25., -11., 15.};
    double expected_x[4] = {1.0,  2.0, -1.0, 1.0};


    double **A ;
    double	*b, *x;

    init2d(&A, n);
    init1d(&b, n);
    init1d(&x, n);

    for(int i = 0; i<n; i++){
		for(int j = 0; j<n; j++){
			A[i][j] = A2[i][j];
		}
	}

    for(int j = 0; j<n; j++){
		b[j] = b2[j];
	}

    print(A, n);
    print(b, n);

    // testing LU
    LUSolve(n, A, b, x);
    print(x, n);

	return 0;
}
