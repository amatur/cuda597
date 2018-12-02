#include <stdio.h>		
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cstring>


#include <mpi.h>
#include "matrix_util.h"
using namespace std;


//~ 
//~ 
//~ double getError(double **A, double* B, double *C, double *x, int Dim){
	//~ matmul(A, x, C, Dim);
	//~ double sum = 0.0;
	//~ for (int i = 0; i<Dim; i++) {
		//~ sum += (C[i] - x[i])*(C[i] - x[i]);
	//~ }
	//~ sum = sqrt(sum);
	//~ return sum;
//~ }



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


int main(int argc, char* argv[]){
	double t_start, t_end, time_secs;
	
  double **A, *A_1d, *b;
  double *X_New, *X_Old, *Bloc_X, tmp;
	
	
    t_start = MPI_Wtime();
   //init mpi 
    MPI_Init(&argc, &argv);
    
    int n = strtol(argv[1], NULL, 10);
    int N = n*n;
	//-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-//
	// Set our tolerance and maximum iterations
	double eps = 1.0e-4;
	int maxit = 2*N*N;


    // get communicator size
    int world_rank, world_size;    
    
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &world_size);
    MPI_Comm_rank(comm, &world_rank);
    
    int Root = 0;
    int &my_rank = world_rank;
    int &num_proc = world_size;

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    //printf("I am (rank %d) of %d size\n\n", world_rank, world_size);

    //starting difficult part

    //if I am process (0,0) I load A, B and gen x
    if (my_rank == 0){
        
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
        
        
        
        
        //printf("x Matrix: ");
        //print(x, N);
        
        
   
        init2d(&A, N);
        fillA_poisson(A, n);
        
        init1d(&b, N);
        
        //printf("A Matrix: \n");
        //print(A, N);
        
        //printf("b Matrix: ");
        //print(b, N);
        //init1d(&x, N);
        
        //fill b
        fillB(b, n);
	
	
          /* ...Convert Matrix_A into 1-D array Input_A ......*/
         A_1d  = (double *)malloc(N*N*sizeof(double));
         convertTo1D(A, A_1d, N);
        //~ int index    = 0;
        //~ for(int irow=0; irow<N; irow++)
            //~ for(int icol=0; icol<N; icol++)
                //~ A_1d[index++] = A[irow][icol];
        
		
        //jacobiSolve(N, A, b, x, eps, maxit);
        
    }

    /* .... Broad cast the size of the matrix to all ....*/
    /*N: dim of mat, 1: 1 integer, Root proc*/
   if (my_rank == Root) {
    // If we are the root process, send our data to everyone
    for (int i = 0; i < num_proc; i++) {
      if (i != my_rank) {
        MPI_Send(&N, 1, MPI_INT, i, 0, comm);
      }
    }
  } else {
    // If we are a receiver process, receive the data from the root
    MPI_Recv(&N, 1, MPI_INT, Root, 0, comm,
            MPI_STATUS_IGNORE);
  }
	
   //~ if (my_rank == Root) {
    //~ // If we are the root process, send our data to everyone
    //~ for (int i = 0; i < num_proc; i++) {
      //~ if (i != my_rank) {
        //~ MPI_Send(&N, 1, MPI_INT, i, 0, comm);
      //~ }
    //~ }
  //~ } else {
    //~ // If we are a receiver process, receive the data from the root
    //~ MPI_Recv(&N, 1, MPI_INT, Root, 0, comm,
            //~ MPI_STATUS_IGNORE);
  //~ }
 


    // check if the number of processor given to us
    // can be appropriately decomposed
    // for that we need num_proc as factor of N
    //if(N % num_proc != 0) {
    //    MPI_Finalize();
    //    if(my_rank == 0){
    //        printf("Matrix Can not be Striped Evenly ..... \n");
    //     }
    //    return 1;
    // }
    int num_rows_block = N/num_proc;
    //cout<<"Broadcast done from"<<my_rank<<"  and nrb"<<num_rows_block<<endl;

    // now the partitioned A, B and x for each of the processor
     /*......Memory of input matrix and vector on each process .....*/
    double *ARecv = (double *) malloc (num_rows_block * N* sizeof(double));
    double *BRecv = (double *) malloc (num_rows_block * sizeof(double));
    
    /*......Scatter the Input Data to all process ......*/
    //if(my_rank == 0){
		MPI_Scatter (A_1d, num_rows_block * N, MPI_DOUBLE, ARecv, num_rows_block * N,
		MPI_DOUBLE, 0, MPI_COMM_WORLD);

		
		 MPI_Scatter (b, num_rows_block, MPI_DOUBLE, BRecv, num_rows_block,
		MPI_DOUBLE, 0, MPI_COMM_WORLD);
		
		//cout<<"Scatter done from"<<my_rank<<"  and nrb"<<num_rows_block<<endl;

	//}


	//initialize auxiliary data structures
	X_New  = (double *) malloc (N * sizeof(double));
	X_Old  = (double *) malloc (N * sizeof(double));
	Bloc_X = (double *) malloc (num_rows_block * sizeof(double));

	/* Initailize X[i] = B[i] */
	for(int irow=0; irow<num_rows_block; irow++){
		Bloc_X[irow] = BRecv[irow];
	}

	/*scatter from all proceess and gather*/
	MPI_Allgather(Bloc_X, num_rows_block, MPI_DOUBLE, X_New, num_rows_block, MPI_DOUBLE, MPI_COMM_WORLD);
	//MPI_Scatter(Bloc_X, num_rows_block, MPI_DOUBLE, X_New, num_rows_block, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	//MPI_Gather(Bloc_X, num_rows_block, MPI_DOUBLE, X_New, num_rows_block, MPI_DOUBLE, my_rank, MPI_COMM_WORLD);


	int GlobalRowNo, index;
  int Iteration = 0;
  do{

		
	   for(int irow=0; irow<N; irow++)
			 X_Old[irow] = X_New[irow];

      for(int irow=0; irow<num_rows_block; irow++){

			
          GlobalRowNo = (my_rank * num_rows_block) + irow;
			 Bloc_X[irow] = BRecv[irow];
			 index = irow * N;

			
			 for(int icol=0; icol<GlobalRowNo; icol++){
				 Bloc_X[irow] -= X_Old[icol] * ARecv[index + icol];
			 }
			 
			 for(int icol=GlobalRowNo+1; icol<N; icol++){
				 Bloc_X[irow] -= X_Old[icol] * ARecv[index + icol];
			 }
          Bloc_X[irow] = Bloc_X[irow] / ARecv[irow*N + GlobalRowNo];
		}

  		MPI_Allgather(Bloc_X, num_rows_block, MPI_DOUBLE, X_New,
						  num_rows_block, MPI_DOUBLE, MPI_COMM_WORLD);
			//MPI_Scatter(Bloc_X, num_rows_block, MPI_DOUBLE, X_New, num_rows_block, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	//MPI_Gather(Bloc_X, num_rows_block, MPI_DOUBLE, X_New, num_rows_block, MPI_DOUBLE, my_rank, MPI_COMM_WORLD);
      
      Iteration++;
  }while( (Iteration < maxit) && (getError(X_Old, X_New, N) >= eps));
	

	
	//free(X_old);
	//free(Bloc_X);
	

    /*
    if(my_rank==0){
		print(X_New, N);
	}
    */

    //-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-//
	// Run jacobi
 
    // start timer
    
    //MPI_File fh; 

    //MPI_File_open(MPI_COMM_WORLD, "time.out",
    //    MPI_MODE_CREATE|MPI_MODE_WRONLY,
    //    MPI_INFO_NULL, &fh); 
    
    //MPI_File_write(fh, &time_secs, 1, MPI_DOUBLE, MPI_STATUS_IGNORE); 
    //MPI_File_close(&fh);    

    //printf("x Matrix: ");
    //print(x, N);
    
    //end = clock();//time count stops 
    
    //double total_time = ((double) (end - start))/ CLOCKS_PER_SEC ;//calulate total time
    
    //printf("nTime taken: %.20lf seconds.\n", total_time); //in seconds
    
    MPI_Finalize();
    
	t_end = MPI_Wtime();
    time_secs = t_end - t_start;
    //cout<< "Time(sec): "<< time_secs << endl;
    if(my_rank==0)
		cout<< time_secs << endl;
    return 0;
}





