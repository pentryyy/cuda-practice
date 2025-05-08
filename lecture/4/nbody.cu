#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define EPSILON 1e-15

// Ядро для расчета ускорений с использованием разделяемой памяти
__global__ void Acceleration_GPU_Shared(
    float *X, float *Y, float *AX, float *AY, int nt, int N
) {

    extern __shared__ float s_data[];

    float *Xs = s_data;
    float *Ys = &s_data[blockDim.x];
    
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    float ax = 0.0f, ay = 0.0f;
    float xx, yy, rr;

    int sh = (nt - 1) * N;

    // Загрузка данных в разделяемую память
    Xs[threadIdx.x] = X[id + sh];
    Ys[threadIdx.x] = Y[id + sh];
    __syncthreads();

    for (int j = 0; j < blockDim.x; j++) {
        if (j + blockIdx.x * blockDim.x != id) {
            xx = Xs[j] - X[id + sh];
            yy = Ys[j] - Y[id + sh];
            rr = sqrtf(xx * xx + yy * yy);
            if (rr > 0.01f) {
                rr = 10.0f / (rr * rr * rr);
                ax += xx * rr;
                ay += yy * rr;
            }
        }
    }
    AX[id] = ax;
    AY[id] = ay;
}

// Ядро для обновления позиций
__global__ void Position_GPU(
    float *X, float *Y, float *VX, float *VY, float *AX, float *AY, float tau, int nt, int N
) {

    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int sh = (nt - 1) * N;

    X[id + nt * N] = X[id + sh] + VX[id] * tau + AX[id] * tau * tau * 0.5f;
    Y[id + nt * N] = Y[id + sh] + VY[id] * tau + AY[id] * tau * tau * 0.5f;
    VX[id] += AX[id] * tau;
    VY[id] += AY[id] * tau;
}

int main() {
    const int N = 10240;      // Количество частиц
    const int NT = 10;        // Количество временных шагов
    const float tau = 0.001f; // Шаг по времени

    size_t mem_size = N * sizeof(float);
    size_t mem_size_big = NT * N * sizeof(float);

    // Выделение памяти на хосте
    float *hX, *hY, *hVX, *hVY, *hAX, *hAY;
    hX = (float*)malloc(mem_size_big);
    hY = (float*)malloc(mem_size_big);
    hVX = (float*)malloc(mem_size);
    hVY = (float*)malloc(mem_size);
    hAX = (float*)malloc(mem_size);
    hAY = (float*)malloc(mem_size);
    printf("[INFO] Host memory allocated.\n");

    // Инициализация начальных условий
    printf("[INFO] Initializing data...\n");
    for (int j = 0; j < N; j++) {
        float phi = (float)rand() / RAND_MAX * 2 * M_PI;
        hX[j] = cosf(phi) * 1e-4f;
        hY[j] = sinf(phi) * 1e-4f;
        float vv = (hX[j] * hX[j] + hY[j] * hY[j]) * 10.0f;
        hVX[j] = -vv * sinf(phi);
        hVY[j] = vv * cosf(phi);
    }
    printf("[INFO] Data initialized.\n");

    // Выделение памяти на устройстве
    float *dX, *dY, *dVX, *dVY, *dAX, *dAY;
    cudaMalloc((void**)&dX, mem_size_big);
    cudaMalloc((void**)&dY, mem_size_big);
    cudaMalloc((void**)&dVX, mem_size);
    cudaMalloc((void**)&dVY, mem_size);
    cudaMalloc((void**)&dAX, mem_size);
    cudaMalloc((void**)&dAY, mem_size);
    printf("[INFO] Device memory allocated.\n");

    // Копирование данных на устройство
    cudaMemcpy(dX, hX, mem_size_big, cudaMemcpyHostToDevice);
    cudaMemcpy(dY, hY, mem_size_big, cudaMemcpyHostToDevice);
    cudaMemcpy(dVX, hVX, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dVY, hVY, mem_size, cudaMemcpyHostToDevice);
    printf("[INFO] Data copied to device.\n");

    // Настройка параметров запуска
    dim3 threads(BLOCK_SIZE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Таймеры
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float gpuTime;

    // Запуск GPU-расчета
    cudaEventRecord(start);
    for (int j = 1; j < NT; j++) {
        Acceleration_GPU_Shared<<<blocks, threads, 2 * BLOCK_SIZE * sizeof(float)>>>(dX, dY, dAX, dAY, j, N);
        Position_GPU<<<blocks, threads>>>(dX, dY, dVX, dVY, dAX, dAY, tau, j, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);
    printf("[INFO] GPU calculation time: %.2f ms\n", gpuTime);

    // Освобождение памяти
    free(hX); free(hY); free(hVX); free(hVY); free(hAX); free(hAY);
    cudaFree(dX); cudaFree(dY); cudaFree(dVX); cudaFree(dVY); cudaFree(dAX); cudaFree(dAY);
    printf("[INFO] Memory freed.\n");

    return 0;
}
