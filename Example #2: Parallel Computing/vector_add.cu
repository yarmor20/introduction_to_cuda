#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

#define N 10'000'000
#define TREADS_PER_BLOCK 256


__global__ void vector_add(size_t n, int *x, int *y) {
    size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Handling arbitrary vector size
    if (thread_id < n) {
        y[thread_id] = y[thread_id] + x[thread_id];
    }
}


void fill_up_arrays(int n, int *x, int *y) {
    for (size_t i = 0; i < n; i++) {
        x[i] = i;
        y[i] = i;
    }
}


int main() {
    int *x, *y; // host copies of a, b, c
    int *d_x, *d_y; // device copies of a, b, c
    size_t size = N * sizeof(int);

    // Allocate space for device copies of a, b, c
    cudaMalloc(&d_x, size);
    cudaMalloc(&d_y, size);

    // Alloc space for host copies of a, b, c and setup input values
    x = (int*)malloc(size);
    y = (int*)malloc(size);
    fill_up_arrays(N, x, y);

    // Copy inputs to device
    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);

    // Launch add() kernel on GPU
    int grid_size = ((N + TREADS_PER_BLOCK - 1) / TREADS_PER_BLOCK);
    vector_add<<<grid_size, TREADS_PER_BLOCK>>>(N, d_x, d_y);

    // Copy result back to host
    cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_x); cudaFree(d_y);
    free(x); free(y);

    printf("PASSED!");
    return 0;
}
