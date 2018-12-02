#include "jacobi.h"
#include "lu.h"
#include "matrix_util.h"

#include<stdio.h>
#include<math.h>
#include<stdlib.h>


bool EXPECT_NEAR(double v1, double v2, double tol){
	return fabs(v1-v2) <= tol;
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
    LUSolve(n, A, b, x, 1e-10);
    print(x, n);

	bool success = true;
    // checking if all values are correct (up to some error value)
    for (int i = 0; i < n; ++i)
    {
        if (!EXPECT_NEAR(expected_x[i], x[i], 1e-10)){
			
			success = false;
		}
        
    }
    
    if(success){
		printf("PASSED LU TEST!\n");
	}else{
		printf("FAILED LU-test\n");
	}
	
	
	success = true;
	double eps = 1e-4;
    jacobiSolve(n, A, b, x, eps, 2*n*n);
    
    print(A, n);
    print(b, n);
	printf("x Matrix: ");
    print(x, n);
	for (int i = 0; i < n; ++i)
    {
        if (!EXPECT_NEAR(expected_x[i], x[i], eps)){
			printf("%lf \n", x[i]);
			success = false;
		}
        
    }
	if(success){
		printf("PASSED Jacobi TEST!");
	}else{
		printf("FAILED Jacobi-test");
	}
	return 0;
}

