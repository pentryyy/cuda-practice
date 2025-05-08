#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vectorAdd(float *dA, float *dB, float *dC, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) dC[i] = dA[i] + dB[i];
}

int main() {
    const int N = 512 * 50000;
    const int mem_size = N * sizeof(float);
    const int num_streams = 4;

    cudaStream_t streams[num_streams];

    float *hA, *hB, *hC;
    float *dA, *dB, *dC;

    // Выделение pinned-памяти
    cudaHostAlloc((void**)&hA, mem_size, cudaHostAllocDefault);
    cudaHostAlloc((void**)&hB, mem_size, cudaHostAllocDefault);
    cudaHostAlloc((void**)&hC, mem_size, cudaHostAllocDefault);
    printf("[INFO] Host memory allocated.\n");

    // Инициализация данных
    printf("[INFO] Initializing data...\n");
    for (int i = 0; i < N; i++) {
        hA[i] = 1.0f / ((i + 1.0f) * (i + 1.0f));
        hB[i] = expf(1.0f / (i + 1.0f));
    }
    printf("[INFO] Data initialized on host.\n");

    cudaMalloc((void**)&dA, mem_size);
    cudaMalloc((void**)&dB, mem_size);
    cudaMalloc((void**)&dC, mem_size);
    printf("[INFO] Device memory allocated.\n");

    // Создание потоков
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Асинхронные операции
    int chunk_size = N / num_streams;
    for (int i = 0; i < num_streams; i++) {
        int offset = i * chunk_size;
        cudaMemcpyAsync(dA + offset, hA + offset, chunk_size * sizeof(float),
                        cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(dB + offset, hB + offset, chunk_size * sizeof(float),
                        cudaMemcpyHostToDevice, streams[i]);
    }

    int threadsPerBlock = 512;
    int blocksPerGrid = (chunk_size + threadsPerBlock - 1) / threadsPerBlock;
    for (int i = 0; i < num_streams; i++) {
        int offset = i * chunk_size;
        vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>(dA + offset, dB + offset, dC + offset, chunk_size);
    }

    printf("[INFO] Copying results to host...\n");
    for (int i = 0; i < num_streams; i++) {
        int offset = i * chunk_size;
        cudaMemcpyAsync(hC + offset, dC + offset, chunk_size * sizeof(float),
                        cudaMemcpyDeviceToHost, streams[i]);
    }

    // Синхронизация
    cudaDeviceSynchronize();
    printf("[INFO] All operations completed.\n");

    // Освобождение ресурсов
    cudaFreeHost(hA);
    cudaFreeHost(hB);
    cudaFreeHost(hC);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    for (int i = 0; i < num_streams; i++) {
        cudaStreamDestroy(streams[i]);
    }
    printf("[INFO] Memory freed.\n");

    return 0;
}
