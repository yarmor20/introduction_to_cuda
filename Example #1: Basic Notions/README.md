# Basic Notions
## CPU vs. GPU

 CPUs and GPUs are separated platforms with their own memory space. In terminology:
 - CPU is called `host` with its memory (host memory)
 - GPU is called `device` with its memory (device memory)
 
 ## Work Flow
 
 1. Copy input data from CPU memory to GPU memory
 2. Load GPU program and execute, caching data on chip for performance
 3. Copy results from GPU memory to CPU memory

## Hello World! with Device Code

```cuda
__global__ void kernel_func() {
}

int main() {
    kernel_func<<<1, 1>>>();
    printf("Hello World!");
    return 0;
}
```
> In fact, CUDA is an extension to C/C++ language, but as we execute not only host code, but also device code, there are some differences.
- CUDA keyword `__global__` indicates that:
  - `kernel_func()` runs on the device
  - `kernel_func()` is called from the host code

- The whole source code is separetad into host and device components by `nvcc` compiler so that:
  - Device functions, such as `kernel_func()` are processed by NVIDIA compiler, while
  - Host functions, such as `main()` are processed by host compiler (gcc, cl.exe)

- The next syntax `<<<1, 1>>>` mark a call from host code to device code
  - In terminology, it is called - "kernel launch"
  - We will talk about `(1, 1)` parameters as we proceed

## Addition on Device

Let us have the next function:
```cuda
__global__ void add(int *a, int *b, int *c) {
    *c = *a + *b;
}
```
- Note that we use pointers for the variables
- As `add()` runs on device, `a`, `b` and `c` must point to device memory

> As our pointers must point to device memory, we need to allocate memory on GPU

- `cudaMalloc()`, `cudaMemcpy()`, `cudaFree` are used for handling device memory
- They are similar to `malloc()`, `memcpy()`, `free()`

For complete example look at `add.cu`

## Run CUDA Program
```
$> nvcc add.cu -o add
$> ./add
```

## References

- CUDA C/C++ Basics. [NVIDIA Developer](https://www.nvidia.com/docs/IO/116711/sc11-cuda-c-basics.pdf)
- Say Hello to CUDA. [CUDA Tutorial](https://cuda-tutorial.readthedocs.io/en/latest/tutorials/tutorial01/)
