#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024

// Ядро для сложения векторов
__global__ void vectorAdd(int *a, int *b, int *c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;

    // Выделение памяти на хосте
    printf("[CPU] Allocating host memory...\n");
    a = (int*)malloc(N * sizeof(int));
    b = (int*)malloc(N * sizeof(int));
    c = (int*)malloc(N * sizeof(int));

    // Инициализация данных
    printf("[CPU] Initializing data...\n");
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    // Вывод первых 5 элементов
    printf("\n[Initial Data]\n");
    printf("a[0..4]: ");
    for (int i = 0; i < 5; i++) printf("%d ", a[i]);
    printf("\n");

    printf("b[0..4]: ");
    for (int i = 0; i < 5; i++) printf("%d ", b[i]);
    printf("\n\n");

    // Выделение памяти на устройстве
    printf("[CUDA] Allocating GPU memory...\n");
    cudaMalloc((void**)&d_a, N * sizeof(int));
    cudaMalloc((void**)&d_b, N * sizeof(int));
    cudaMalloc((void**)&d_c, N * sizeof(int));

    // Копирование данных на устройство
    printf("[CUDA] Copying data to GPU...\n");
    cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

    // Запуск ядра
    printf("[CUDA] Launching kernel (4 blocks of 256 threads)...\n");
    vectorAdd<<<4, 256>>>(d_a, d_b, d_c);

    // Копирование результата обратно
    printf("[CUDA] Copying result to CPU...\n");
    cudaMemcpy(c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Проверка ошибок
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("\n[CUDA ERROR]: %s\n", cudaGetErrorString(err));
    }

    // Проверка результата
    printf("\n[Results]\n");
    printf("c[0..4]: ");
    for (int i = 0; i < 5; i++) printf("%d ", c[i]);
    printf("\n");

    // Проверка всех элементов
    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (c[i] != a[i] + b[i]) {
            errors++;
            if (errors <= 3) {
                printf("Error at c[%d]: %d != %d\n", i, c[i], a[i] + b[i]);
            }
        }
    }

    if (errors > 0) {
        printf("\nTotal errors: %d\n", errors);
    } else {
        printf("\nAll elements correct!\n");
    }

    // Освобождение памяти
    printf("\n[Cleanup] Freeing memory...\n");
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a);
    free(b);
    free(c);

    return 0;
}
