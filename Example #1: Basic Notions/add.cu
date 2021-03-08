
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>


__global__ void add(int *c, int *a, int *b) {
    *c = *a + *b;
}

int main() {
    int a = 5, b = 4, c; // host copies of a, b, c
    int *d_a, *d_b, *d_c; // device copies of a, b, c

    // Allocate space for device copies of a, b, c
    cudaMalloc(&d_a, sizeof(int));
    cudaMalloc(&d_b, sizeof(int));
    cudaMalloc(&d_c, sizeof(int));

    // Copy inputs to device
    cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);

    // Launch add() kernel on GPU
    add<<<1, 1>>>(d_c, d_a, d_b);

    // Copy result back to host
    cudaMemcpy(&c, d_c, sizeof(int), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    printf("Result: %d\n", c);
    return 0;
}

