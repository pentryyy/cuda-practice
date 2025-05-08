#include <stdio.h>
#include <cuda_runtime.h>

#define N 2048
#define BLOCK_SIZE 32

__global__ void matrixMul(float *A, float *B, float *C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    if (row < N && col < N) {
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    
    size_t size = N * N * sizeof(float);

    // Выделение памяти на хосте
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);
    printf("[INFO] Host memory allocated.\n");

    // Инициализация данных
    printf("[INFO] Initializing matrices...\n");
    for (int i = 0; i < N * N; i++) {
        h_A[i] = (float)(i % N);
        h_B[i] = (float)(i / N);
    }
    printf("[INFO] Data initialized.\n");

    // Выделение памяти на устройстве
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);
    printf("[INFO] Device memory allocated.\n");

    // Копирование данных на устройство
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    printf("[INFO] Data copied to device.\n");

    // Запуск ядра
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(N / BLOCK_SIZE, N / BLOCK_SIZE);
    printf("[INFO] Launching kernel with %dx%d blocks and %dx%d threads.\n", blocks.x, blocks.y, threads.x, threads.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matrixMul<<<blocks, threads>>>(d_A, d_B, d_C);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpuTime;
    cudaEventElapsedTime(&gpuTime, start, stop);
    printf("[INFO] GPU calculation time: %.2f ms\n", gpuTime);

    // Копирование результатов обратно
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    printf("[INFO] Results copied to host.\n");

    // Освобождение памяти
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    printf("[INFO] Memory freed.\n");

    return 0;
}
