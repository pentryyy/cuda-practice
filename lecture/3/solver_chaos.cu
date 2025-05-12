#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define N 2048
#define EPS_G 1e-6
#define EPS_L 1e-15

// Ядро для вычисления матрицы A(x)
__global__ void matrixA(double *dA, double *dX, int size) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < size && j < size) {
        dA[i + j * size] = pow(sin(dX[j]) * cos(dX[i]), 2.0) + (i == j ? (double)size : 0.0);
    }
}

int main() {
    double *hX, *hF, *hDelta;
    double *dX, *dF, *dA, *dDelta;

    const int mem_sizeX = N * sizeof(double);
    const int mem_sizeA = N * N * sizeof(double);
    
    const dim3 threadsPerBlock(16, 16);
    const dim3 blocksPerGrid((N + 15) / 16, (N + 15) / 16);

    // Выделение pinned-памяти
    cudaHostAlloc((void**)&hX, mem_sizeX, cudaHostAllocDefault);
    cudaHostAlloc((void**)&hF, mem_sizeX, cudaHostAllocDefault);
    cudaHostAlloc((void**)&hDelta, mem_sizeX, cudaHostAllocDefault);
    printf("[INFO] Pinned memory allocated on host.\n");

    // Инициализация данных
    printf("[INFO] Initializing data...\n");
    for (int i = 0; i < N; i++) {
        hX[i] = 1.0;
        hF[i] = 0.5;
    }

    // Выделение памяти на устройстве
    cudaMalloc((void**)&dX, mem_sizeX);
    cudaMalloc((void**)&dF, mem_sizeX);
    cudaMalloc((void**)&dA, mem_sizeA);
    cudaMalloc((void**)&dDelta, mem_sizeX);
    printf("[INFO] Device memory allocated.\n");

    // Асинхронное копирование данных
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaMemcpyAsync(dX, hX, mem_sizeX, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dF, hF, mem_sizeX, cudaMemcpyHostToDevice, stream);
    printf("[INFO] Data copied asynchronously.\n");

    // Вычисление матрицы A
    matrixA<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(dA, dX, N);
    printf("[INFO] Matrix_A kernel launched.\n");

    // Синхронизация и завершение
    cudaStreamSynchronize(stream);
    printf("[INFO] All operations completed.\n");

    // Освобождение ресурсов
    cudaFreeHost(hX);
    cudaFreeHost(hF);
    cudaFreeHost(hDelta);
    cudaFree(dX);
    cudaFree(dF);
    cudaFree(dA);
    cudaFree(dDelta);
    cudaStreamDestroy(stream);
    printf("[INFO] Memory freed.\n");

    return 0;
}
