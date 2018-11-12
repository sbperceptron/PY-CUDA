
import numpy as np
import time
from pycuda import driver, compiler, gpuarray, tools
import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import numpy
from pycuda.compiler import SourceModule


mod= SourceModule("""
	// Kernel definition
	__global__ void Vecmul(float* A, float* B, float* C)
{
    int i = threadIdx.x;
    C[i] = A[i] * B[i];
}

""")

st1=time.time()
a=np.random.randn(1,400).astype(np.float32)
b=np.random.randn(1,400).astype(np.float32)
c=numpy.empty_like(a)
a_gpu=cuda.mem_alloc(a.nbytes)
b_gpu=cuda.mem_alloc(b.nbytes)
c_gpu=cuda.mem_alloc(c.nbytes)

# then transfer the data to Gpu
cuda.memcpy_htod(a_gpu,a)
cuda.memcpy_htod(b_gpu,b)

# ////////////////////////////////
# kernel invocation with N threads
func=mod.get_function("Vecmul")
func(a_gpu,b_gpu,c_gpu, block=(1,4,1))

mul=numpy.empty_like(a)
cuda.memcpy_dtoh(mul, c_gpu)
ed1=time.time()

st2=time.time()
prod=a*b
ed2=time.time()
print "GPU: ",np.round((ed1-st1),7)
print "CPU: ",np.round((ed2-st2),7)
