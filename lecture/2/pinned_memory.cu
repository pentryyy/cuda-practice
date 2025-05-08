#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vectorAdd(float *dA, float *dB, float *dC, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) dC[i] = dA[i] + dB[i];
}

int main() {
    const int N = 512 * 50000;
    const int mem_size = N * sizeof(float);

    float *hA, *hB, *hC;
    float *dA, *dB, *dC;

    // Использование pinned-памяти
    cudaHostAlloc((void**)&hA, mem_size, cudaHostAllocDefault);
    cudaHostAlloc((void**)&hB, mem_size, cudaHostAllocDefault);
    cudaHostAlloc((void**)&hC, mem_size, cudaHostAllocDefault);

    // Инициализация данных
    printf("[INFO] Initializing data...\n");
    for (int i = 0; i < N; i++) {
        hA[i] = 1.0f / ((i + 1.0f) * (i + 1.0f));
        hB[i] = expf(1.0f / (i + 1.0f));
    }
    printf("[INFO] Data initialized.\n");

    cudaMalloc((void**)&dA, mem_size);
    cudaMalloc((void**)&dB, mem_size);
    cudaMalloc((void**)&dC, mem_size);

    cudaMemcpy(dA, hA, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, mem_size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 512;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    printf("[INFO] Kernel launched.\n");
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(dA, dB, dC, N);

    printf("[INFO] Copying results...\n");
    cudaMemcpy(hC, dC, mem_size, cudaMemcpyDeviceToHost);

    // Освобождение pinned-памяти
    cudaFreeHost(hA);
    cudaFreeHost(hB);
    cudaFreeHost(hC);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    printf("[INFO] Memory and streams released.\n");

    return 0;
}
