#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32
#define N 2048

// Ядро с оптимизацией через разделяемую память и устранением банковских конфликтов
__global__ void matmulOptimized(float *A, float *B, float *C) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE + 1];

    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;

    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;
    
    float sum = 0.0f;

    for (int k = 0; k < N; k += BLOCK_SIZE) {
        As[ty][tx] = A[row * N + (k + tx)];
        Bs[ty][tx] = B[(k + ty) * N + col];
        __syncthreads();

        for (int i = 0; i < BLOCK_SIZE; i++) {
            sum += As[ty][i] * Bs[i][tx];
        }
        __syncthreads();
    }

    C[row * N + col] = sum;
}

int main() {
    size_t size = N * N * sizeof(float);
    
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

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

    // Выделение памяти на устройстве
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);
    printf("[INFO] Device memory allocated.\n");

    // Копирование данных на устройство
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    printf("[INFO] Data copied to device.\n");

    // Настройка параметров запуска
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(N / BLOCK_SIZE, N / BLOCK_SIZE);

    // Таймер
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float gpuTime;

    // Запуск ядра
    cudaEventRecord(start);
    matmulOptimized<<<blocks, threads>>>(d_A, d_B, d_C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
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
