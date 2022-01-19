import faulthandler

faulthandler.enable()  # Shows page faults, generates exception when using dill/pickle (disable in this case)

import timeit
import random
import numpy
from numba import float32, vectorize, uint32, void, guvectorize

# PyCUDA
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.compiler as compiler
from pycuda.compiler import SourceModule

NumberOfIndividuals = 1000
MAX_INDIVIDUALS = (1 << 15)
BLOCK_SIZE = 1024
MAX_IND_PER_HH = 5

mod = SourceModule("""
// Test compile with "nvcc --cubin -arch sm_61 -m64 -Ie:\\src\\ifdm\\multiscale\\venv\\lib\\site-packages\\pycuda\\cuda kernel.cu"

#include <cstdint>

    #define NSTATES (1<<14)
    #define MAX_HOUSEHOLDS (10000)
    #define MAX_IND_PER_HH  (5)
    __device__ int updated_households[MAX_HOUSEHOLDS][MAX_IND_PER_HH];
    __device__ unsigned int index;

    __global__ void update( int* households, int* last_index )
    {
        int idx = threadIdx.x + blockIdx.x*blockDim.x;  // calculate array index

        if(idx >= MAX_HOUSEHOLDS) return;      
        
        if( updated_households[idx][0] != -1 )
        {            
            const int resultIndex = atomicAdd(&index, 1); // increase index by 1
            for(int i = 0; i < MAX_IND_PER_HH; ++i)
            {
                households[resultIndex * MAX_IND_PER_HH + i] = updated_households[idx][i];
                updated_households[idx][i] = -1;    // clear
            }
            __syncthreads();
            last_index[0] = atomicAdd(&index, 1);   // last_index gets overwritten several times with same value
        }          
    }
    
    __global__ void init( )
    {
        int idx = threadIdx.x + blockIdx.x*blockDim.x;  // calculate array index

        if(idx >= MAX_HOUSEHOLDS) return;
        
        if ( threadIdx.x == 0 )
        {
            index = 0;
        }
        
        for(int i = 0; i < MAX_IND_PER_HH; ++i)
        {
            updated_households[idx][i] = -1;
        }                                 
    }
    
    __global__ void get_initial(int* households)
    {
        int idx = threadIdx.x + blockIdx.x*blockDim.x;  // calculate array index
        
        for(int i = 0; i < MAX_IND_PER_HH; ++i)
        {
            households[idx * MAX_IND_PER_HH + i] = updated_households[idx][i];
        } 
    }
    
    __global__ void fill_with_same_random_values()
    {
        int idx = threadIdx.x + blockIdx.x*blockDim.x;  // calculate array index

        if( idx == 0 || idx == 2 || idx == 3 || idx == 4 || idx > 7  || idx >= MAX_HOUSEHOLDS ) return;
        
        for(int i = 0; i < MAX_IND_PER_HH -2; ++i)
        {
            updated_households[idx][i] = i*idx;
        }                                 
    }
""")


MAX_HOUSEHOLDS = 10000

_next_id = MAX_HOUSEHOLDS
#_updated_households = pycuda.driver.managed_zeros(shape=(MAX_HOUSEHOLDS, MAX_IND_PER_HH), dtype=numpy.int32, mem_flags=cuda.mem_attach_flags.GLOBAL)
_initial_households = numpy.ones(shape=(MAX_HOUSEHOLDS, MAX_IND_PER_HH), dtype=numpy.int32) * -1
_updated_households = numpy.ones(shape=(MAX_HOUSEHOLDS, MAX_IND_PER_HH), dtype=numpy.int32) * -1
#_updated_households[:] = -1
#_last_index = pycuda.driver.managed_zeros(shape=1, dtype=numpy.int32, mem_flags=cuda.mem_attach_flags.GLOBAL)
_last_index = numpy.zeros(shape=1, dtype=numpy.int32)

def get_hh():
    grid_x = (_next_id + BLOCK_SIZE - 1) // BLOCK_SIZE
    gpu_get_fn = mod.get_function("get_initial")
    initial_households_on_gpu = cuda.mem_alloc(_initial_households.size * _initial_households.dtype.itemsize)  # allocate memory on gpu
    gpu_get_fn(initial_households_on_gpu, block=(BLOCK_SIZE, 1, 1), grid=(grid_x, 1), time_kernel=True)
    cuda.memcpy_dtoh(_initial_households, initial_households_on_gpu)

def init():
    grid_x = (_next_id + BLOCK_SIZE - 1) // BLOCK_SIZE
    gpu_init_fn = mod.get_function("init")
    gpu_init_fn(block=(BLOCK_SIZE, 1, 1), grid=(grid_x, 1), time_kernel=True)

def fill():
    grid_x = (_next_id + BLOCK_SIZE - 1) // BLOCK_SIZE
    gpu_fill_fn = mod.get_function("fill_with_same_random_values")
    gpu_fill_fn(block=(BLOCK_SIZE, 1, 1), grid=(grid_x, 1), time_kernel=True)

def update():
    grid_x = (_next_id + BLOCK_SIZE - 1) // BLOCK_SIZE
    gpu_update_fn = mod.get_function("update")

    updated_households_on_gpu = cuda.mem_alloc(_updated_households.size * _updated_households.dtype.itemsize)  # allocate memory on gpu
    last_index_on_gpu = cuda.mem_alloc(_last_index.size * _last_index.dtype.itemsize)  # allocate memory on gpu for result

    cuda.memcpy_htod(updated_households_on_gpu, _updated_households)
    cuda.memcpy_htod(last_index_on_gpu, _last_index)

    gpu_update_fn(updated_households_on_gpu, last_index_on_gpu, block=(BLOCK_SIZE, 1, 1), grid=(grid_x, 1), time_kernel=True)

    cuda.memcpy_dtoh(_updated_households, updated_households_on_gpu)
    cuda.memcpy_dtoh(_last_index, last_index_on_gpu)


if __name__ == '__main__':
    init()
#     fill()
#     get_hh()
#
#     print("_initial_households 0:10")
#     for p in _initial_households[0:10]:
#         print(p)
# #    update()
#     print("_updated_households 0:10")
    # for p in _updated_households[0:10]:
    #     print(p)
    # print(_last_index)
    #
    # get_hh()
    # print("_initial_households 0:10")
    # for p in _initial_households[0:10]:
    #     print(p)