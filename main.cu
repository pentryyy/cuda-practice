#include <iostream>
#include <cuda_runtime.h>

__global__ void vectorAdd(const float* A, const float* B, float* C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    const int numElements = 1024;
    size_t size = numElements * sizeof(float);

    // Выделение памяти на хосте
    float* h_A = new float[numElements];
    float* h_B = new float[numElements];
    float* h_C = new float[numElements];

    // Инициализация данных
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }

    // Выделение памяти на устройстве
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Копирование данных на устройство
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Запуск ядра
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

    // Копирование результатов обратно
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Проверка результатов
    bool error = false;
    for (int i = 0; i < numElements; ++i) {
        if (fabs(h_C[i] - (h_A[i] + h_B[i])) > 1e-5) {
            error = true;
            break;
        }
    }
    
    std::cout << (error ? "Error!" : "Success!") << std::endl;

    // Освобождение памяти
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}