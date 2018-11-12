# PY-CUDA
pycuda gives you easy access to nvidias cuda parallel cumputation api.

meta programming
In meta programming one writes a program that writes another program that accomplishes a task

# Why metaprogramming?

# Automated tuning
1. what is the optimal number of threads per block?
2. How much data should i work on at once?
3. What data should be loaded into shared memory and how big should the correponding blocks be?

We need to come up with a heuristic that will allow us to reliably pick the fastest version
Which is very hard given the variety of hardware generations
So PYCUDA solves the problem by benchmarking at the runtime and usse whatever works fastest

this is a important advantage of pycuda over cuda: since it lets you make these decisions while your code is running

# Data Types
the code needs to deal with different data types at run time. 
it may for example have to work on both single and double precision floating point
numbers.

But here we can generate whatever code that is needed right when it is needed

# specialize cide for the given problem
generate code for the problem you are being asked to solve 
instead to keep code unnecessarily generic

# constants are faster than Variables
if the problem size vary from run to run, but you perform a larger number 
of kernel invocaeions on data of identical size, you may want to cinsider compiling data size into 
your cide as a constant . this can have significant performance benefits. resulting mainly
from decreased fetch times and less register pressure in particular, 
multiplications by constants are much more efficiently carried out than general 
variable variable multiplications.

# loop unrolling
with metaprogramming, you can dynamically unroll your loops to 
the needed size in Python.

# CUDA-notes
cuda notes
# Kernels
cuda c extends c by allowing the programmer to define c functions, called kernels, 
that, when called are excecuted n times in parallel by n different cuda kernels, 
as opposed to only once like regular c functions.

a kernel is defined using __global__ declaration specifier and the number of cuda 
threads that execute that kernel for a given kernel call is specified using a new
<<<...>>> excecution configuration syntax. each thread that ececutes the kernel is 
given a unique thread id that is accessible within the kernel through the built in 
threadIdx variable

the following code add two vectors a and b of size n and stores the result into a 
vector c
!["Add_vectors"](https://github.com/sbperceptron/CUDA-notes/blob/master/add.c)

here each of the N threads that excecute VecAdd() performs one pair wise addition

# Thread hierarchy 
For convenience threadIdx is a 3 component vector, so that threads can be identified using
a one dimensional, two dimensional or three dimensional thread index, forming a one 
dimensional, two dimensional or three dimensional blick of threads called thread blick
this provides a natural wat to invoke computation across the elements in a domain
such as a vector, matrix, or volume.

the index of a thread and its thread id relate to each other in astraight forward way
. For a one dimensional block, they are the same ; for a two dimensional block of 
size Dx, Dy the thread id of a thread of index x,y is x+yDx; for a three dimensional block og
size Dx, Dy , Dz the thread id of a thread of index x,y,z is x+yDx+zDxDy

for example the code below adds two matricesa A and B of size NXN and stores the 
result into matrix c

!["Add_ Matrices"](https://github.com/sbperceptron/CUDA-notes/blob/master/addmat.c)

there is a limit to the number of threads per block, since all threads of a block 
are expected to reside on the same processor core and must share the limited memory
resources of that core. on current gpus a thread block may contain up to 1024 threads.

however, a kernel can be executed by multiple equally shaped thread blocks, so thaat 
the total number of threads is equal to the number of threads per block times the number of 
blocks.

blocks are organized into a one dimensional, two dimensional or three dimensional 
grid of thread blocks

!["Thread Blocks"](https://github.com/sbperceptron/CUDA-notes/blob/master/grid-of-thread-blocks.png)

each block within the grid can be identified by a one dimensional , two dimensioanl or 
three dimensional index accessible within the kernek through the built-in blockIdx
variable. the dimesnion of the thred block is accessible within the kernel through the builtin
blockDim variable.

 

Given the heterogeneous nature of the cuda programming model, a typical sequence 
of operations for a cuda c program is
1. Declare and allocate host and device memory. 
2. initialize host data.
3. transfer data from the host to device
4. excecute one or more kernels.
5. transfer results from the device to the host.

saxpy single precision a*x plus y 
it is a good hello world example for parallel computaion

the function saxpy is the function that runs in parallel on gpu
and the main function is the host code
the main function declares two pairs of arrays

*float *x, *y, *d_x, *d_y;\n
 x = (float*)malloc(N*sizeof(float));
 y = (float*)malloc(N*sizeof(float));

 cudaMalloc(&d_x, N*sizeof(float)); 
 cudaMalloc(&d_y, N*sizeof(float));*

# compiling a cuda c file
*nvcc -o saxpy saxpy.cu*
