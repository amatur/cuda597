#include <stdio.h>
#include <time.h> 
//#include <conio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

#define N 10

__global__ void Suma_vec( int *a, int *b, int *c, int n )
{
int tid = threadIdx.x; // Identificador del thread
c[tid] = a[tid] + b[tid];
}

int main(void)
{
int A[N], B[N], C[N];
int *dA, *dB, *dC;
srand (time(NULL)); 

//Se crea el vector A
for(int i=0; i<N; i++)
A[i] = rand() % 101; 

//Se crea la matriz B
for(int i=0; i<N; i++)
B[i] = rand() % 101; 

//Se reserva memoria en la GPU
cudaMalloc( (void**)&dA, N * sizeof(int)); 
cudaMalloc( (void**)&dB, N * sizeof(int)); 
cudaMalloc( (void**)&dC, N * sizeof(int)); 

//Se copian los vectores A y B en la GPU
cudaMemcpy( dA, A, N * sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy( dB, B, N * sizeof(int), cudaMemcpyHostToDevice);

Suma_vec<<<1,N>>>(dA, dB, dC, N);

//Se copia el resultado obtenido (GPU) en el vector C de la CPU
cudaMemcpy( C, dC, N * sizeof(int), cudaMemcpyDeviceToHost);

for (int j=0; j<N; j++)
{
cout<<A[j]<<"\t"<<B[j]<<"\t"<<C[j]<<endl; 
}

cudaFree( dA);
cudaFree( dB);
cudaFree( dC);
cudaDeviceSynchronize();
cudaError_t error = cudaGetLastError();
if(error!=cudaSuccess)
{
   fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
   exit(-1);
}
//getch();
return 0;
}
