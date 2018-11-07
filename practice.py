import pycuda.autoinit
import pycuda.driver as cuda
import numpy
from pycuda.compiler import SourceModule

# transfering the data onto device
a=numpy.random.randn(4,4)
# a consist of double precision numbers but most nvidia devices only support 
# single precision
a=a.astype(numpy.float32)
# we need to transfer data, so we need to allocate memory on the device
a_gpu =cuda.mem_alloc(a.nbytes)
# then transfer the data to Gpu
cuda.memcpy_htod(a_gpu,a)

# for this tutorial we will write cide to double esch entry in a gpu 
# for this we will write the coressponding cuda c code and feed it tinto the 
# cinstuctor of a pycuda.compiler.sorcemodule

mod= SourceModule("""
	__global__ void doublify(float *a)
	{
	int idx = threadIdx.x +threadIdx.y*4;
	a[idx] *=2;
	}
	""")
# The code is now loaded into device 

func= mod.get_function("doublify")
func(a_gpu, block=(4,4,1))

# finally we will get the data back from the Gpu and display it 
a_doubled=numpy.empty_like(a)
cuda.memcpy_dtoh(a_doubled, a_gpu)
print a_doubled
print a

# pycuda does the automatic cleanup
# shortcuts to explicit mem copy
# pycusa.driver.in, pucuda.driver.out , and pycrda.ddriver.inout argument 
# handlrs can simplify some of the memory transfers. for example instead of 
# creating a_gpu, if replacing a is fine, the following code can be used

# func(cuda.InOut(a), block=(4,4,1))

# prepared invocations
# function invocations using the built in pycuda.driver.Function.__call__() 
# mthod incird overhesd for type identification . to schieve the same effect
# as above without this overhesd, the function is bound to argument types and
# then called. this also avoids having to assign explicit argument sizes using 
# the numpy.mumber classes
# grid=(1,1)
# block=(4,4,1)
# func.prepare("P")
# func.prepared_call(grid,block, a_gpu)

# abstracting away the complications
# using a pucuda.gpuarray.GPUArray, the same effect can be achieved with much 
# less writing
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import numpy

a_gpu = gpuarray.to_gpu(numpy.random.randn(4,4).astype(numpy.float32))
a_doubled=(2*a_gpu).get()
print a_doubled
print a 


# advanced topics 
# stuctures
# suppose we have the folloeing structure for doubling a number of variable 
# length arrays

mod = SourceModule("""
    struct DoubleOperation {
        int datalen, __padding; // so 64-bit ptrs can be aligned
        float *ptr;
    };

    __global__ void double_array(DoubleOperation *a) {
        a = &a[blockIdx.x];
        for (int idx = threadIdx.x; idx < a->datalen; idx += blockDim.x) {
            a->ptr[idx] *= 2;
        }
    }
    """)

# here each block in the grid will double one of the arrays. The for loop allowa
# for more data elements than threds to be doubled, thoughe is not efficient 
# if one can guarantee that there will be a suficient number of threads. Nect,
# a wrapper class for the structure is created, and two arrays are instantiated

class DoubleOpStruct:
    mem_size = 8 + numpy.intp(0).nbytes
    def __init__(self, array, struct_arr_ptr):
        self.data = cuda.to_device(array)
        self.shape, self.dtype = array.shape, array.dtype
        cuda.memcpy_htod(int(struct_arr_ptr), numpy.getbuffer(numpy.int32(array.size)))
        cuda.memcpy_htod(int(struct_arr_ptr) + 8, numpy.getbuffer(numpy.intp(int(self.data))))
    def __str__(self):
        return str(cuda.from_device(self.data, self.shape, self.dtype))

struct_arr = cuda.mem_alloc(2 * DoubleOpStruct.mem_size)
do2_ptr = int(struct_arr) + DoubleOpStruct.mem_size

array1 = DoubleOpStruct(numpy.array([1, 2, 3], dtype=numpy.float32), struct_arr)
array2 = DoubleOpStruct(numpy.array([0, 4], dtype=numpy.float32), do2_ptr)
print("original arrays", array1, array2)

# this cide uses pycuda.driver.to_device() and pycuda.driver,from_device() 
# functions to allocate and copy values, and demonstates how offsets to an allocated 
# block of memory can be used. finally, the cide can be executed; the following 
# demonstates doubling both arrays, then only the secind: 

func = mod.get_function("double_array")
func(struct_arr, block = (32, 1, 1), grid=(2, 1))
print("doubled arrays", array1, array2)

func(numpy.intp(do2_ptr), block = (32, 1, 1), grid=(1, 1))
print("doubled second only", array1, array2, "\n")

