
import numpy as np
import time
from pycuda import driver, compiler, gpuarray, tools
import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import numpy
from pycuda.compiler import SourceModule


# thread idx is a 3 component vector so threads can be identified as a one dimensional, two dimensioal and 
# three dimensional index. which means we can form thread blocks of one ,two and three dimensions

# index of a thread and its thread id are related as follows 
# for a one dimensional block 
# the thread index is same as thread index
# for a two dimensional block of shape Dx Dy
# thread id of a thread of index x,y is 
# x+y*Dx
# for a three dimensional block of size Dx, Dy, Dz 
# the thread id of a thread od index is 
# x+y*Dx+z*Dx*Dy

mod= SourceModule("""
	// Kernel definition
	__global__ void Vecmul(float* A, float* B, float* C)
{
    int i = threadIdx.x;
    int j = threadIdx.y;
    C[i][j] = A[i][j] * B[i][j];
}

""")

st1=time.time()
a=np.random.randn(4,4).astype(np.float32)
b=np.random.randn(4,4).astype(np.float32)
c=numpy.empty_like(a)
a_gpu=cuda.mem_alloc(a.nbytes)
b_gpu=cuda.mem_alloc(b.nbytes)
c_gpu=cuda.mem_alloc(c.nbytes)

# then transfer the data to Gpu
cuda.memcpy_htod(a_gpu,a)
cuda.memcpy_htod(b_gpu,b)

# ////////////////////////////////
# // Kernel invocation with one block of N * N * 1 threads
func=mod.get_function("Vecmul")
func(a_gpu,b_gpu,c_gpu, block=(4,4,1))

mul=numpy.empty_like(a)
cuda.memcpy_dtoh(mul, c_gpu)
ed1=time.time()

st2=time.time()
prod=a*b
ed2=time.time()
print "GPU: ",np.round((ed1-st1),7)
print "CPU: ",np.round((ed2-st2),7)
