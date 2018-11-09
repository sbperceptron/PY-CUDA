#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" 
Multiplies two square matrices together using a *single* block of threads and 
global memory only. Each thread computes one element of the resulting matrix.
"""

import numpy as np
import time
from pycuda import driver, compiler, gpuarray, tools

import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import numpy
from pycuda.compiler import SourceModule
import time

gpu_start=time.time()
# transfering the data onto device
a=numpy.random.randn(4000,4000)
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
    a[idx] *=a[idx];
    }
    """)
# The code is now loaded into device 

func= mod.get_function("doublify")
func(a_gpu, block=(4,4,1))

# finally we will get the data back from the Gpu and display it 
a_doubled=numpy.empty_like(a)
cuda.memcpy_dtoh(a_doubled, a_gpu)

gpu_end=time.time()
print "GPU Time:", np.round((gpu_end - gpu_start),4)

cpu_start=time.time()
b=a*a
cpu_end=time.time()
print "CPU Time:", np.round((cpu_end - cpu_start),4)
