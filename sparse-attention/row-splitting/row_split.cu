#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>

void row_split_spmm(const float *A_dense, const float *B_dense, int m, int k, int n) {
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

    // invoke kernel

}
