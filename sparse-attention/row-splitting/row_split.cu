#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include "../utils/cycleTimer.h"

#define DEVICE 0
#define MAX_SEQ_LEN 512

static const char *_cusparseGetErrorEnum(cusparseStatus_t error)
{
    switch (error)
    {

        case CUSPARSE_STATUS_SUCCESS:
            return"CUSPARSE_STATUS_SUCCESS";

        case CUSPARSE_STATUS_NOT_INITIALIZED:
            return"CUSPARSE_STATUS_NOT_INITIALIZED";

        case CUSPARSE_STATUS_ALLOC_FAILED:
            return"CUSPARSE_STATUS_ALLOC_FAILED";

        case CUSPARSE_STATUS_INVALID_VALUE:
            return"CUSPARSE_STATUS_INVALID_VALUE";

        case CUSPARSE_STATUS_ARCH_MISMATCH:
            return"CUSPARSE_STATUS_ARCH_MISMATCH";

        case CUSPARSE_STATUS_MAPPING_ERROR:
            return"CUSPARSE_STATUS_MAPPING_ERROR";

        case CUSPARSE_STATUS_EXECUTION_FAILED:
            return"CUSPARSE_STATUS_EXECUTION_FAILED";

        case CUSPARSE_STATUS_INTERNAL_ERROR:
            return"CUSPARSE_STATUS_INTERNAL_ERROR";

        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            return"CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";

        case CUSPARSE_STATUS_ZERO_PIVOT:
            return"CUSPARSE_STATUS_ZERO_PIVOT";

        case CUSPARSE_STATUS_NOT_SUPPORTED:
            return"CUSPARSE_STATUS_NOT_SUPPORTED";
    }

    return"<unknown>";
}

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void __cusparseSafeCall(cusparseStatus_t err, const char *file, const int line)
{
    if (CUSPARSE_STATUS_SUCCESS != err) {
        fprintf(stderr,"CUSPARSE error in file '%s', line %d, error %s terminating!", __FILE__, __LINE__, _cusparseGetErrorEnum(err));
        assert(0);
    }
}
extern"C" void cusparseSafeCall(cusparseStatus_t err) { __cusparseSafeCall(err, __FILE__, __LINE__); }

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

void row_split_spmm(const float *h_A_dense, const float *h_B_dense, int m, int k, int n) {
    int val;
    cudaDeviceGetAttribute(&val, cudaDevAttrPageableMemoryAccess, DEVICE);
    float *A_dense, *B_dense, *C_dense;
    gpuErrchk(cudaMalloc(&A_dense, m * k * sizeof(float)));
    gpuErrchk(cudaMalloc(&B_dense, k * n * sizeof(float)));
    gpuErrchk(cudaMemcpy(A_dense, h_A_dense, m * k * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(B_dense, h_B_dense, k * n * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc(&C_dense, m * n * sizeof(float)));
    // --- Initialize cuSPARSE

    cusparseHandle_t handle;
    cusparseSafeCall(cusparseCreate(&handle));
    // Initialize matrix descriptors
    cusparseMatDescr_t descrA;
    cusparseSafeCall(cusparseCreateMatDescr(&descrA));
    cusparseSafeCall(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));

    // Get nnz's
    int nnzA = 0;
    int *nnzPerVectorA;
    const int lda = m;
    gpuErrchk(cudaMalloc(&nnzPerVectorA, k * sizeof(*nnzPerVectorA)));
    cusparseSafeCall(cusparseSnnz(handle, CUSPARSE_DIRECTION_ROW, m, k, descrA, A_dense, lda, nnzPerVectorA, &nnzA));

    // declare CSR data
    float *A_data;
    gpuErrchk(cudaMalloc(&A_data, nnzA * sizeof(*A_data)));

    // declare CSR row-pointers & col indices
    int *A_row_ptrs, *A_col_indices;
    gpuErrchk(cudaMalloc(&A_row_ptrs, (m + 1) * sizeof(*A_row_ptrs)));
    gpuErrchk(cudaMalloc(&A_col_indices, nnzA * sizeof(*A_col_indices)));

    // fill CSR arrays
    cusparseSafeCall(cusparseSdense2csr(handle, m, k, descrA, A_dense, lda, nnzPerVectorA, A_data, A_row_ptrs,
                                        A_col_indices));

    // invoke kernel
    dim3 blockDim(32);   // each
    // TODO: reduce grid size using nnz
    dim3 gridDim(get_grid_len(m, blockDim.x), get_grid_len(n, blockDim.y));
    double start = CycleTimer::currentSeconds();
    device_spmm<<<gridDim, blockDim>>>(m, k, n, A_data, A_row_ptrs, A_col_indices, B_dense, C_dense);
    gpuErrchk(cudaDeviceSynchronize());
    double end = CycleTimer::currentSeconds();
    printf("row-spliting matmul:    %.4f ms\n", 1000.f * (end - start));

}
