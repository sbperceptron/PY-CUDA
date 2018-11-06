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
