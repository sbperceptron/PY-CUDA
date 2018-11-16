import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import numpy as np
from pycuda.compiler import SourceModule
import time
# numpy arrays are passed to kernels as pointers to flat arrays
# the blocks size for carrying out the computation is at max 32*32 or the max
# number of threads possible per block 1024
# Blocks are organized into a one-dimensional, two-dimensional, or three-dimensional 
# grid of thread blocks
# the number of threads in a thread block are dictated by the size of the data
# being processed or the number of processors in the system
# each block in the grid can be identified by a one dimesional, two dimensional
# three dimensional index accesible within the kernel through the built in
############################# blockidx ###################################
# the dimension of the theread block is accessible through
############################# blockdim ###################################

# the code to handle multiple thread blocks

# the performance gains on using gpu over cpu are not observed. it is observed the
# cpu is able to pull of a better performance compared to the gpu.
# this is attributed to the fact that gpu computation involves moving the data
# to the gpu memory and after the computation transfering it back. so,
# the drastic difference in the performance bw cpu and gpu can be seen
# when the gpu is able to do more computations on the same data

kernel_code_template = """__global__ void MatAdd(float *A, float *B, float *C)
{	
	// 3D Thread ID (assuming that more than one block will be executed)
    // int tx = threadIdx.x;
    // int ty = threadIdx.y;
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    // Pvalue is used to store the element of the matrix
    // that is computed by the thread
    float Pvalue = 0;

    // Each thread loads one row of M and one column of N, 
    //   to produce one element of P.
    if (tx < %(N)s && ty < %(N)s){
    for (int k = 0; k < %(N)s; ++k) {
        float Aelement = A[ty * %(N)s + k];
        float Belement = B[k * %(N)s + tx];
        Pvalue = Aelement + Belement;
        }

    // Write the matrix to device memory;
    // each thread writes one element
    C[ty * %(N)s + tx] = Pvalue;
    }

}"""


N = 16
a = np.ones((N,N), dtype=np.float32)
b = np.ones((N,N), dtype=np.float32)

s1=time.time()
c_cpu = a*b
e1=time.time()


d_a = gpuarray.to_gpu(a) 
d_b = gpuarray.to_gpu(b) 

# create empty gpu array for the result (C = A * B)
d_c = gpuarray.empty((N,N), np.float32)

kernel_code = kernel_code_template % {
      'N': N 
    }

s2=time.time()
mod=SourceModule(kernel_code)
func= mod.get_function("MatAdd")
func(d_a, d_b, d_c, grid = (N // 16, N // 16), block=(16,16,1))

e2=time.time()

print d_c.get()
print "Gpu time:", np.round((e2-s2),4)
print "cpu time:", np.round((e1-s1),4)
print np.allclose(c_cpu, d_c.get())