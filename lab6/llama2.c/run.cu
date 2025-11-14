#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <assert.h>

static float *g_d_x = NULL;      // GPU 输入向量缓冲区
static float *g_d_xout = NULL;   // GPU 输出向量缓冲区
static size_t g_max_n = 0;       // 当前缓冲区支持的最大 n
static size_t g_max_d = 0;       // 当前缓冲区支持的最大 d

static cublasHandle_t g_handle = NULL;
static float* g_d_weights = NULL;     // GPU 上的 weights 基址
static float* g_h_weights = NULL;     // Host 上的 weights 基址
static size_t g_weights_bytes = 0;

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS Error at %s:%d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#ifdef __cplusplus
extern "C" {
#endif

// 把 mmap 后的权重基址一次性拷贝到 GPU，创建 cuBLAS 句柄
void init_cuda_weights(float* h_weights_base, size_t bytes) {
    if (g_d_weights != NULL) {
        return;
    }
    if (h_weights_base == NULL || bytes == 0) return;
    g_h_weights = h_weights_base;
    g_weights_bytes = bytes;
    CUBLAS_CHECK(cublasCreate(&g_handle));
    CUDA_CHECK(cudaMalloc((void**)&g_d_weights, g_weights_bytes));
    CUDA_CHECK(cudaMemcpy(g_d_weights, g_h_weights, g_weights_bytes, cudaMemcpyHostToDevice));
}

void init_cuda_buffers(size_t max_n, size_t max_d) {
    // 如果已有足够大的缓冲区，无需重新分配
    if (g_max_n >= max_n && g_max_d >= max_d) {
        return;
    }

    // 释放旧缓冲区
    if (g_d_x) {
        cudaFree(g_d_x);
        g_d_x = NULL;
    }
    if (g_d_xout) {
        cudaFree(g_d_xout);
        g_d_xout = NULL;
    }

    // 分配新缓冲区
    CUDA_CHECK(cudaMalloc((void**)&g_d_x, max_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&g_d_xout, max_d * sizeof(float)));

    g_max_n = max_n;
    g_max_d = max_d;
}

void free_cuda_weights() {
    if (g_d_weights) {
        CUDA_CHECK(cudaFree(g_d_weights));
        g_d_weights = NULL;
    }
    if (g_handle) {
        CUBLAS_CHECK(cublasDestroy(g_handle));
        g_handle = NULL;
    }
    g_h_weights = NULL;
    g_weights_bytes = 0;

    // 释放缓冲区
    if (g_d_x) {
        cudaFree(g_d_x);
        g_d_x = NULL;
    }
    if (g_d_xout) {
        cudaFree(g_d_xout);
        g_d_xout = NULL;
    }
    g_max_n = g_max_d = 0;
}

void matmul_cuda(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // 检查缓冲区是否足够
    if ((size_t)n > g_max_n || (size_t)d > g_max_d) {
        // 动态扩展
        init_cuda_buffers((size_t)n, (size_t)d);
    }

    float *d_w = NULL;

    if (g_d_weights != NULL && g_h_weights != NULL) {
        ptrdiff_t elem_offset = w - g_h_weights;
        if (elem_offset < 0) {
            d_w = NULL;
        } else {
            d_w = g_d_weights + elem_offset;
        }
    }
    if(d_w == NULL) {
        assert("Weights not initialized in GPU memory!" && 0);
    }


    CUDA_CHECK(cudaMemcpy(g_d_x, x, (size_t)n * sizeof(float), cudaMemcpyHostToDevice));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    CUBLAS_CHECK(cublasSgemv(g_handle, CUBLAS_OP_T, n, d, &alpha, d_w, n, g_d_x, 1, &beta, g_d_xout, 1));

    CUDA_CHECK(cudaMemcpy(xout, g_d_xout, (size_t)d * sizeof(float), cudaMemcpyDeviceToHost));
}

#ifdef __cplusplus
}
#endif