# Parallel Computing on GPU

The key to parallel computing is CUDA's `<<<1, 1>>>` syntax.
> This is called the execution configuration, and it tells the CUDA runtime how many parallel threads to use for the launch on the GPU.

Apparently, there are two parameters:
- Number of thread `blocks`
- Number of `threads` in each block
> CUDA GPUs run kernels using blocks of threads that are a multiple of 32 in size, so 256 threads is a reasonable size to choose.

## Working with Threads

Let us look at the next function call:
```cuda
vector_add<<<1, 256>>>(N, x, y);
```
> This code will do the computation once per thread, rather than spreading the computation across the parallel threads.

To do it properly, we'll need to change our kernel function `vector_add()`. CUDA C++ provides keywords that let kernels get the indices of the running threads. 
- `threadIdx.x` contains the index of the current thread within its block
- `blockDim.x` contains the number of threads in the block.

After modifying our function, it will look like:
```cuda
__global__ void vector_add(size_t *n, int *x, int *y) {
    size_t thread_idx = threadIdx.x;
    
    if (thread_idx < n) {
        y[thread_idx] = y[thread_idx] + x[thread_idx];
    }
}
```

## Including Thread Blocks
> CUDA GPUs have many parallel processors grouped into Streaming Multiprocessors, or SMs. 
> Each SM can run multiple concurrent thread blocks. 
> As an example, a Tesla P100 GPU based on the Pascal GPU Architecture has 56 SMs, each capable of supporting up to 2048 active threads. 

To take full advantage of all these threads, we should launch the kernel with multiple thread blocks.

> Together, the blocks of parallel threads make up what is known as the grid.

If we have, for example, `N` elements and `256` threads per block, we would like to engage at least `N` threads. To do so, we will use the next formula:
```cuda
/* Being careful to round up numbers that are not multiple of TREADS_PER_BLOCK */
int block_num = ((N + TREADS_PER_BLOCK - 1) / TREADS_PER_BLOCK);
vector_add<<<block_num, TREADS_PER_BLOCK>>>(N, x, y);
```

Now, we also need to update kernel code. CUDA C++ provides keywords:
- `gridDim.x`, which contains the number of blocks in the grid
- `blockIdx.x`, which contains the index of the current thread block in the grid

```cuda
__global__ void vector_add(size_t *n, int *x, int *y) {
    size_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (thread_idx < n) {
        y[thread_idx] = y[thread_idx] + x[thread_idx];
    }
}
```

## Benchmarks

The next benchmarks where taken from [NVIDIA Developer Blog](https://developer.nvidia.com/blog/even-easier-introduction-cuda/), using laptop videocard `GeForce GTX 750M`.

Version | Time | Bandwidth | Speedup
------- | ---- | --------- | -------
1 Thread | 411ms | 30.6 MB/s | 1.0x
1 CUDA Block | 3.2ms | 3.9 GB/s | 128.0x
Many CUDA Blocks | 0.68ms | 18.5 GB/s | 604.0x

## References

- An Even Easier Introduction to CUDA. [NVIDIA Developer Blog](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)
- CUDA C/C++ Basics. [NVIDIA Developer](https://www.nvidia.com/docs/IO/116711/sc11-cuda-c-basics.pdf)
- Constraints on Threads and Blocks. [StackOverflow](https://stackoverflow.com/questions/9985912/how-do-i-choose-grid-and-block-dimensions-for-cuda-kernels?noredirect=1&lq=1)
