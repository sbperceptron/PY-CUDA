#include <stdio.h>
// Kernel definition
//adds two vectors A and B of size N and stores the result into vector C: 

__global__ void VecAdd(float* A, float* B, float *C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i] ;
}

int main()
{
    // here only 1024 is the maximum number i can use for carrying out the 
    // additions that is the max number of threads i can have per thread block

    int N = 1024;

    float *A, *B, *C, *d_A, *d_B, *d_C;
  	A = (float*)malloc(N*sizeof(float));
  	B = (float*)malloc(N*sizeof(float));
    C = (float*)malloc(N*sizeof(float));


  	cudaMalloc(&d_A, N*sizeof(float)); 
  	cudaMalloc(&d_B, N*sizeof(float));
    cudaMalloc(&d_C, N*sizeof(float));

  	for (int i = 0; i < N; i++) {
    A[i] = 1.0f;
    B[i] = 2.0f;
    C[i] = 0.0f;
  	}

  	cudaMemcpy(d_A, A, N*sizeof(float), cudaMemcpyHostToDevice);
  	cudaMemcpy(d_B, B, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, N*sizeof(float), cudaMemcpyHostToDevice);

    // Kernel invocation with N threads
    VecAdd<<<1, N>>>(d_A, d_B, d_C);

    
    cudaMemcpy(C, d_C, N*sizeof(float), cudaMemcpyDeviceToHost);

    float sum = 0.0f;
    for (int i = 0; i < N; i++){
      sum = sum + C[i];
    }
    printf("Sum is: %f\n", sum); 

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  free(A);
  free(B);
  free(C);
    
}