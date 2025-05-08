#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#define N (512 * 50000)
#define BLOCK_SIZE 512

__global__ void processArrays(float *dA, float *dB, float *dC, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        float ab = dA[i] * dB[i];
        float sum = 0.0f;
        for (int j = 0; j < 100; j++) {
            sum += sinf(j + ab);
        }
        dC[i] = sum;
    }
}

int main() {
    float *hA, *hB, *hC;
    float *dA, *dB, *dC;
    
    size_t mem_size = N * sizeof(float);

    // Выделение памяти на хосте
    hA = (float*)malloc(mem_size);
    hB = (float*)malloc(mem_size);
    hC = (float*)malloc(mem_size);
    printf("[INFO] Host memory allocated.\n");

    // Инициализация данных
    printf("[INFO] Initializing data...\n");
    for (int i = 0; i < N; i++) {
        hA[i] = sinf(i);
        hB[i] = cosf(2 * i - 5);
    }
    printf("[INFO] Data initialized.\n");

    // Выделение памяти на устройстве
    cudaMalloc((void**)&dA, mem_size);
    cudaMalloc((void**)&dB, mem_size);
    cudaMalloc((void**)&dC, mem_size);
    printf("[INFO] Device memory allocated.\n");

    // Копирование данных на устройство
    cudaMemcpy(dA, hA, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, mem_size, cudaMemcpyHostToDevice);
    printf("[INFO] Data copied to device.\n");

    // Запуск ядра
    dim3 threadsPerBlock(BLOCK_SIZE);
    dim3 blocksPerGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    printf("[INFO] Launching kernel with %d blocks and %d threads per block.\n", blocksPerGrid.x, threadsPerBlock.x);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    processArrays<<<blocksPerGrid, threadsPerBlock>>>(dA, dB, dC, N);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpuTime;
    cudaEventElapsedTime(&gpuTime, start, stop);
    printf("[INFO] GPU calculation time: %.2f ms\n", gpuTime);

    // Копирование результатов обратно
    cudaMemcpy(hC, dC, mem_size, cudaMemcpyDeviceToHost);
    printf("[INFO] Results copied to host.\n");

    // Освобождение памяти
    free(hA); free(hB); free(hC);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    printf("[INFO] Memory freed.\n");

    return 0;
}
