import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import numpy as np
from pycuda.compiler import SourceModule

N=1024
a= np.ones(N).astype(np.float32)
b= np.ones(N).astype(np.float32)
c= np.zeros(N).astype(np.float32)

d_a =cuda.mem_alloc(a.nbytes)
d_b =cuda.mem_alloc(b.nbytes)
d_c =cuda.mem_alloc(c.nbytes)

cuda.memcpy_htod(d_a,a)
cuda.memcpy_htod(d_b,b)
cuda.memcpy_htod(d_c,c)

mod=SourceModule("""__global__ void VecAdd(float *A, float *B, float *C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i] ;
}""")

func= mod.get_function("VecAdd")
func(d_a,d_b,d_c, block=(N,1,1))

vecsum=np.empty_like(a)
cuda.memcpy_dtoh(vecsum, d_c)
print vecsum