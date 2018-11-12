// Kernel definition
__global__ void MatAdd(float A[N][N], float B[N][N],
float C[N][N])
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N)
        C[i][j] = A[i][j] + B[i][j];
}

int main()
{
    ...
    // Kernel invocation
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
    ...
}

// the code to handle multiple blocks
// the grid is created with enough blocks to have one thread per matrix
// element 
// and also we assume that the threads per grid is evenly divisible
// by the number of threads per block in that dimension

// thread blocks are required to execute independently: it must be possible 
// to excecute them in any order parallel or in series. this independence requrement 
// allows thread blocks to be scheduled in any order across any number of cores

// threads within thread blocks can cooperate by sharing data through shared
// memory and by synchronising the execution to coordinate memory access. 
