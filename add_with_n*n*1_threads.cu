// Kernel definition
__global__ void MatAdd(float *A, float *B,
                       float *C)
{
    int i = threadIdx.x;
    int j = threadIdx.y;
    C[i][j] = A[i][j] + B[i][j];
}

int main()
{
    int N = 10;
    float *A, *B, *C, *d_A, *d_B, *d_C;
    
    A = (float*)malloc(N*sizeof(float));
    B = (float*)malloc(N*sizeof(float));
    C = (float*)malloc(N*sizeof(float));


    cudaMalloc(&d_A, N*sizeof(float)); 
    cudaMalloc(&d_B, N*sizeof(float));
    cudaMalloc(&d_C, N*sizeof(float));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++){
            A[i][j] = 1.0f;
            B[i][j] = 2.0f;
            C[i][j] = 0.0f;
        }
    }

    cudaMemcpy(d_A, A, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, N*sizeof(float), cudaMemcpyHostToDevice);

    // Kernel invocation with one block of N * N * 1 threads
    int numBlocks = 1;
    dim3 threadsPerBlock(N, N);
    MatAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C);
}