#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define EPS 1e-15
#define N 10240

// Ядро для решения СЛАУ методом итераций
__global__ void Solve(double *dA, double *dF, double *dX0, double *dX1, int size) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < size) {
        double aa, sum = 0.0;
        for (int j = 0; j < size; j++) {
            sum += dA[j + t * size] * dX0[j];
            if (j == t) aa = dA[j + t * size];
        }
        dX1[t] = dX0[t] + (dF[t] - sum) / aa;
    }
}

// Ядро для вычисления погрешности
__global__ void Eps(double *dX0, double *dX1, double *delta, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        delta[i] = fabs(dX0[i] - dX1[i]);
        dX0[i] = dX1[i];
    }
}

int main() {
    double *hA, *hF, *hX0, *hX1, *hDelta;
    double *dA, *dF, *dX0, *dX1, *delta;
    
    const int mem_sizeA = N * N * sizeof(double);
    const int mem_sizeX = N * sizeof(double);
    const int threadsPerBlock = 512;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Выделение памяти на хосте
    hA = (double*)malloc(mem_sizeA);
    hF = (double*)malloc(mem_sizeX);
    hX0 = (double*)malloc(mem_sizeX);
    hX1 = (double*)malloc(mem_sizeX);
    hDelta = (double*)malloc(mem_sizeX);
    printf("[INFO] Host memory allocated.\n");

    // Инициализация данных (примерная)
    printf("[INFO] Initializing data...\n");
    for (int i = 0; i < N; i++) {
        hF[i] = 1.0;
        hX0[i] = 0.0;
        for (int j = 0; j < N; j++) {
            hA[j + i * N] = (i == j) ? 2.0 : 0.1; // Диагональное преобладание
        }
    }
    printf("[INFO] Data initialized.\n");

    // Выделение памяти на устройстве
    cudaMalloc((void**)&dA, mem_sizeA);
    cudaMalloc((void**)&dF, mem_sizeX);
    cudaMalloc((void**)&dX0, mem_sizeX);
    cudaMalloc((void**)&dX1, mem_sizeX);
    cudaMalloc((void**)&delta, mem_sizeX);
    printf("[INFO] Device memory allocated.\n");

    // Копирование данных на устройство
    cudaMemcpy(dA, hA, mem_sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(dF, hF, mem_sizeX, cudaMemcpyHostToDevice);
    cudaMemcpy(dX0, hX0, mem_sizeX, cudaMemcpyHostToDevice);
    printf("[INFO] Data copied to device.\n");

    // Итерационный процесс
    double eps = 1.0;
    int iter = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    printf("[INFO] Starting iterations...\n");
    while (eps > EPS) {
        Solve<<<blocksPerGrid, threadsPerBlock>>>(dA, dF, dX0, dX1, N);
        Eps<<<blocksPerGrid, threadsPerBlock>>>(dX0, dX1, delta, N);
        cudaMemcpy(hDelta, delta, mem_sizeX, cudaMemcpyDeviceToHost);

        eps = 0.0;
        for (int i = 0; i < N; i++) eps += hDelta[i];
        eps /= N;
        iter++;
        printf("[INFO] Iteration %d: eps = %e\n", iter, eps);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float timerValueGPU;
    cudaEventElapsedTime(&timerValueGPU, start, stop);
    printf("[INFO] GPU calculation time: %.2f ms\n", timerValueGPU);

    // Освобождение памяти
    free(hA);
    free(hF);
    free(hX0);
    free(hX1);
    free(hDelta);
    cudaFree(dA);
    cudaFree(dF);
    cudaFree(dX0);
    cudaFree(dX1);
    cudaFree(delta);
    printf("[INFO] Memory freed.\n");

    return 0;
}
