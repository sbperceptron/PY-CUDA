import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import numpy as np
from pycuda.compiler import SourceModule

# numpy arrays are passed to kernels as pointers to flat arrays
kernel_code_template = """__global__ void MatAdd(float *A, float *B, float *C)
{	
	// 2D Thread ID (assuming that only *one* block will be executed)
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Pvalue is used to store the element of the matrix
    // that is computed by the thread
    float Pvalue = 0;

    // Each thread loads one row of M and one column of N, 
    //   to produce one element of P.
    for (int k = 0; k < %(N)s; ++k) {
        float Aelement = A[ty * %(N)s + k];
        float Belement = B[k * %(N)s + tx];
        Pvalue += Aelement + Belement;
        }

    // Write the matrix to device memory;
    // each thread writes one element
    C[ty * %(N)s + tx] = Pvalue;

}"""

N=15
a= np.ones((N,N)).astype(np.float32)
b= np.ones((N,N)).astype(np.float32)


d_a = gpuarray.to_gpu(a) 
d_b = gpuarray.to_gpu(b) 

# create empty gpu array for the result (C = A * B)
d_c = gpuarray.empty((N,N), np.float32)

kernel_code = kernel_code_template % {
      'N': N 
    }

mod=SourceModule(kernel_code)

func= mod.get_function("MatAdd")
func(d_a, d_b, d_c, block=(N,N,1))

print d_c.get()

np.allclose(c_cpu, c_gpu.get())