#include <iostream>
#include <vector>
#include <cmath>


#define N 2048
#define THREADS 32
#define BLOCKS (N / THREADS)

#define SHARED_MEM_SIZE 2048


__global__ void matrixMul(const int *a, const int *b, int *c, size_t N) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    // Allocate shared (cache) memory
    __shared__ int s_a[SHARED_MEM_SIZE];
    __shared__ int s_b[SHARED_MEM_SIZE];

    /* Shared (L1 Cache) memory capacity is very limited, so in order not
     * to get over its boundaries we will divide our multiplication on tiles.
     *
     * In other words, let us have two vectors:
     *    a1 = [2, 4, 3, 1]
     *    b1 = [2, 4, 3, 1]T (where T means Transposed)
     * We divide these vectors into 2 tiles, [2, 4], [3, 1] and [2, 4]T, [3, 1]T
     * Then we load these first tiles of vectors A and B
     * ([2, 4] and [2, 4]T) into s_a and s_b respectively,
     * and then do the multiplication [2, 4] * [2, 4]T that we be written in temp. */

    int temp = 0
    for (size_t tile = 0; tile < N; tile += blockDim.x) {
        // Every single thread will load a single element there
        s_a[threadIdx.y * blockDim.x + threadIdx.x] = a[row * N + i + threadIdx.x];
        s_b[threadIdx.y * blockDim.x + threadIdx.x] = b[i * N + threadIdx.y * N + col];

        __syncthreads();

        for (int k = 0; k < blockDim.x; k++) {
            // Accumulate results for a single element
            temp += s_a[threadIdx.y * blockDim.x + k] * s_b[k * blockDim.x + threadIdx.x];
        }
        __syncthreads();
    }
    c[row * N + col] = temp;
}


void matrixMulHost(const int *a, const int *b, int *c, size_t N) {
    for (size_t row_a = 0; row_a < N; row_a++) {
        for (size_t col_b = 0; col_b < N; col_b++) {
            for (size_t pair = 0; pair < N; pair++) {
                c[row_a * N + col_b] += a[row_a * N + pair] * b[pair * N + col_b];
            }
        }
    }
}


int main() {
    int *a, *b, *c;
    size_t size = N * N * sizeof(int);

    // Allocate host memory
    malloc(a, size);
    malloc(b, size);
    malloc(c, size);

    // Initialize matrices
    std::generate(a.begin(), a.end(), []() { return rand() % 100; });
    std::generate(b.begin(), b.end(), []() { return rand() % 100; });

    // Allocate device memory
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Copy matrices to device memory
    cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), size, cudaMemcpyHostToDevice);

    // dim3 struct is used for grids (as we have a 2D matrices)
    dim3 thread_grid(THREADS, THREADS);
    dim3 block_grid(BLOCKS, BLOCKS);

    // Rum matrix multiplication on device
    matrixMul<<<block_grid, thread_grid>>>(d_a, d_b, d_c, N);

    // Copy the data back to the host
    cudaMemcpy(c.data(), d_c, size, cudaMemcpyDeviceToHost);

    // Free memory on device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free memory on host
    free(a);
    free(b);
    free(c);

    return 0;
}
