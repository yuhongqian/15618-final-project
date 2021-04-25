#include <cublas_v2.h>

// C(m,n) = A(m,k) * B(k,n)
void gpu_blas_mmul(const float *h_sparse, const float *h_dense, const int m, const int k, const int n) {
    float *d_sparse, *d_dense, *d_res;
    cudaMalloc(&d_sparse, m * k * sizeof(float));
    cudaMalloc(&d_dense, k * n * sizeof(float));
    cudaMalloc(&d_res, m * n * sizeof(float));
    cudaMemcpy(d_sparse, h_sparse, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dense, h_dense, k * n * sizeof(float), cudaMemcpyHostToDevice);

    // invoke cublas function
    int lda=m,ldb=k,ldc=m;
    const float alf = 1;
    const float bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;
    // Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
    // Do the actual multiplication
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, d_sparse, lda, d_dense, ldb, beta, d_res, ldc);
    // Destroy the handle
    cublasDestroy(handle);

    // copy results back to host
}