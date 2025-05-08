#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void complexKernel(float *dA, float *dB, float *dC, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float ab = dA[i] * dB[i];
        float sum = 0.0f;
        for (int j = 0; j < 100; j++) {
            sum += sinf(ab + j);
        }
        dC[i] = sum;
    }
}

int main() {
    const int N = 512 * 50000;
    const int mem_size = N * sizeof(float);

    float *hA, *hB, *hC;
    float *dA, *dB, *dC;

    cudaHostAlloc((void**)&hA, mem_size, cudaHostAllocDefault);
    cudaHostAlloc((void**)&hB, mem_size, cudaHostAllocDefault);
    cudaHostAlloc((void**)&hC, mem_size, cudaHostAllocDefault);

    // Инициализация данных
    for (int i = 0; i < N; i++) {
        hA[i] = sinf(i);
        hB[i] = cosf(2 * i - 5);
    }
    printf("[INFO] Initializing data...\n");

    cudaMalloc((void**)&dA, mem_size);
    cudaMalloc((void**)&dB, mem_size);
    cudaMalloc((void**)&dC, mem_size);

    cudaMemcpy(dA, hA, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, mem_size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 512;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    printf("[INFO] Kernel launched.\n");
    complexKernel<<<blocksPerGrid, threadsPerBlock>>>(dA, dB, dC, N);

    printf("[INFO] Copying results...\n");
    cudaMemcpy(hC, dC, mem_size, cudaMemcpyDeviceToHost);

    // Освобождение памяти
    cudaFreeHost(hA);
    cudaFreeHost(hB);
    cudaFreeHost(hC);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    printf("[INFO] Memory and streams released.\n");

    return 0;
}
