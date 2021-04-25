#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include "../utils/cycleTimer.h"

#define DEVICE 0
#define MAX_SEQ_LEN 128

__device__ __inline__ int device_get_idx(int row, int col, int width) {
    return row * width + col;
}

// Currently, each thread is responsible for one output element
__global__ void device_spmm(int m, int k, int n, float *A_data, int *A_row_ptrs, int *A_col_indices,
                            const float *B_dense, float *C_dense) {
    int m_idx = blockIdx.y;
    int n_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (m_idx > m || n_idx > n) return;

    // TODO: reduce shared memory size
    // Load the corresponding row from A into shared mem
    __shared__ float curr_row[MAX_SEQ_LEN];
    __shared__ float curr_col_idxs[MAX_SEQ_LEN];

    // n < k, so each thread may be responsible for loading multiple row elements from A into shared mem
    int row_start = A_row_ptrs[m_idx];
    int row_end = A_row_ptrs[m_idx + 1];
    int row_nnz = row_end - row_start;
    for (int i = n_idx; i < row_nnz; i += blockDim.x) {
        curr_row[i] = A_data[row_start + i];
        curr_col_idxs[i] = A_col_indices[row_start + i];
    }
    __syncthreads();

    // Each thread loops through the corresponding col in B
    float res = 0;
    for (int i = 0; i < row_nnz; i++) {
        float elem = curr_row[i] * curr_col_idxs[i];
        res += curr_row[i] * elem;
    }
    C_dense[device_get_idx(m_idx, n_idx, n)];
}

int get_grid_len(int number_elems, int block_dim) {
    return (number_elems + block_dim - 1) / block_dim;
}

void row_split_spmm(const float *A_dense, const float *B_dense, int m, int k, int n) {

    float *C_dense;
    cudaMallocManaged(&C_dense, m * n * sizeof(float));
    // --- Initialize cuSPARSE
    cusparseHandle_t handle;
    cusparseCreate(&handle);

    // Initialize matrix descriptors
    cusparseMatDescr_t descrA;
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

    // Get nnz's
    int nnzA = 0;
    int *nnzPerVectorA;
    const int lda = m;
    cudaMallocManaged(&nnzPerVectorA, k * sizeof(*nnzPerVectorA));
    cusparseSnnz(handle, CUSPARSE_DIRECTION_ROW, m, k, descrA, A_dense, lda, nnzPerVectorA, &nnzA);

    // declare CSR data
    float *A_data;
    cudaMallocManaged(&A_data, nnzA * sizeof(*A_data));

    // declare CSR row-pointers & col indices
    int *A_row_ptrs, *A_col_indices;
    cudaMallocManaged(&A_row_ptrs, (m + 1) * sizeof(*A_row_ptrs));
    cudaMallocManaged(&A_col_indices, nnzA * sizeof(*A_col_indices));

    // fill CSR arrays
    cusparseSdense2csr(handle, m, k, descrA, A_dense, lda, nnzPerVectorA, A_data, A_row_ptrs, A_col_indices);

    // prefetch memory to avoid kernel runtime page fault
    cudaMemPrefetchAsync(&A_data, nnzA * sizeof(*A_data), DEVICE);
    cudaMemPrefetchAsync(&A_row_ptrs, (m + 1) * sizeof(*A_row_ptrs), DEVICE);
    cudaMemPrefetchAsync(&A_col_indices, nnzA * sizeof(*A_col_indices), DEVICE);
    cudaMemPrefetchAsync(&B_dense, k * n * sizeof(*B_dense), DEVICE);

    // invoke kernel
    dim3 blockDim(32);   // each
    // TODO: reduce grid size using nnz
    dim3 gridDim(get_grid_len(m, blockDim.x), get_grid_len(n, blockDim.y));
    double start = CycleTimer::currentSeconds();
    device_spmm<<<gridDim, blockDim>>>(m, k, n, A_data, A_row_ptrs, A_col_indices, B_dense, C_dense);
    double end = CycleTimer::currentSeconds();
    printf("row-spliting matmul:    %.4f ms\n", 1000.f * (end - start));

}
