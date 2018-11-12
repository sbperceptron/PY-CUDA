#include <stdio.h>
// the kernels such as saxpy are defined using __global__ declaration
// specifier. 
// variables defined with in the device code do not need to be specified as 
// device variables as they are presumed to reside on the device
// in this case the variables n a and i are stored by each thread on 
// registers. and the pointers x and y are pointers to the device memory 
// the first two arguments n and a are not explicitly transfered to device in
// host code this is because function arguments are passed by value by default
// in c or c++ and cuda can automatically handle the transfer of the values

// as we know the kernel is excecuted by multiple threads in parallel.
// but if we want each thread to process a element of resultant array then 
// we need a means of distinguishing each and identifying each thread.
// cuda defines blockdim, blockidx and threadidx. these predefined variables // are of the type dim3.

// the predefined variable blockdim contains the dimension of each thread 
// block

// the predefined variables thread idx and the blockidx contain the index of // thread with in thread block and the thread block with in the grid

// so the index i is the global index that is used to access the elements of // the arrays. 

// griddim is another parameter that stores the dimensions of the grid

// the second line performs the saxpy operation element wisse


__global__
void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

int main(void)
{
  int N = 1<<20;

  // the pointers x and y point to the host arrays allocated with malloc in // the typical fashion
  // the arrats dx and dy point to device arrays allocated with cuda malloc
  // the host and device in cuda has different memory spaces, both of which // can be managed from host code 
  float *x, *y, *d_x, *d_y;
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));

  cudaMalloc(&d_x, N*sizeof(float)); 
  cudaMalloc(&d_y, N*sizeof(float));

  // the host code that inits the host arrays 
  // x is array of ones and y is an array of 2s

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // to init the device arrays copy the data from x,y to dx and dy
  // using cudamemcpy. the fourth argument indicates the direction of copy

  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

  // Perform SAXPY on 1M elements
  // the info between the triple chevrons dictates the number of device 
  // threads excecute the kernel in parallel. In cuda programming we call the 
  // group of thread blocks launched are called grid of thread blocks

  // the first argument specifies the number of thread blocks in the grid
  // the second argument specifies the number of threads in the block

  // thread blocks and grids can be made one, two and three dimensional by 
  // passing dim3 values for these arguments
  // this example we neeed one dimension so only integers are passed
  // so this case have 256 threads and use the arithmetic division to 
  // determine the number of blocks 

  saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);

  // after running the kernel, to get results back we copy the results from 
  // device array dy to y using the cudamemxcpy
  cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = max(maxError, abs(y[i]-4.0f));
  printf("Max error: %f\n", maxError);

  // after done processing, we have to free the memory allocated. for device // memory allocated using cudamalloc use cudafree and use free for host memory

  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);
}


// compiling and running the code. we use the cuda c nvcc compiler to compile // the code in a file with .cu extension
nvcc -o saxpy saxpy.cu