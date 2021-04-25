#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>

#include <cuda_runtime.h>
#include <cusparse_v2.h>

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

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

inline void __cusparseSafeCall(cusparseStatus_t err, const char *file, const int line)
{
    if (CUSPARSE_STATUS_SUCCESS != err) {
        fprintf(stderr,"CUSPARSE error in file '%s', line %d, error %s terminating!", __FILE__, __LINE__, _cusparseGetErrorEnum(err));
        assert(0);
    }
}


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
extern"C" void cusparseSafeCall(cusparseStatus_t err) { __cusparseSafeCall(err, __FILE__, __LINE__); }


/********/
/* MAIN */
/********/


void cusparse_mmul(const float *h_A_dense, const float *h_B_dense, int m, int k, int n)
{
    // --- Initialize cuSPARSE
    cusparseHandle_t handle;
    cusparseSafeCall(cusparseCreate(&handle));

    /**************************/
    /* SETTING UP THE PROBLEM */
    /**************************/

    float *h_C_dense = (float*)malloc(m * n * sizeof(*h_C_dense));

    // --- Create device arrays and copy host arrays to them
    float *d_A_dense;  gpuErrchk(cudaMalloc(&d_A_dense, m * k * sizeof(*d_A_dense)));
    float *d_B_dense;  gpuErrchk(cudaMalloc(&d_B_dense, k * n * sizeof(*d_B_dense)));
    float *d_C_dense;  gpuErrchk(cudaMalloc(&d_C_dense, m * n * sizeof(*d_C_dense)));
    gpuErrchk(cudaMemcpy(d_A_dense, h_A_dense, m * k * sizeof(*d_A_dense), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_B_dense, h_B_dense, k * n * sizeof(*d_B_dense), cudaMemcpyHostToDevice));

    // --- Descriptor for sparse matrix A
    cusparseMatDescr_t descrA;
    cusparseSafeCall(cusparseCreateMatDescr(&descrA));
    cusparseSafeCall(cusparseSetMatType     (descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    cusparseSafeCall(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE));

    // --- Descriptor for sparse matrix B
    cusparseMatDescr_t descrB;
    cusparseSafeCall(cusparseCreateMatDescr(&descrB));
    cusparseSafeCall(cusparseSetMatType     (descrB, CUSPARSE_MATRIX_TYPE_GENERAL));
    cusparseSafeCall(cusparseSetMatIndexBase(descrB, CUSPARSE_INDEX_BASE_ONE));

    // --- Descriptor for sparse matrix C
    cusparseMatDescr_t descrC;
    cusparseSafeCall(cusparseCreateMatDescr(&descrC));
    cusparseSafeCall(cusparseSetMatType     (descrC, CUSPARSE_MATRIX_TYPE_GENERAL));
    cusparseSafeCall(cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ONE));

    int nnzA = 0;                           // --- Number of nonzero elements in dense matrix A
    int nnzB = 0;                           // --- Number of nonzero elements in dense matrix B

    const int lda = m;                      // --- Leading dimension of dense matrix

    // --- Device side number of nonzero elements per row of matrix A
    // TODO: is the size of d_nnzPerVectorA correct?
    int *d_nnzPerVectorA;   gpuErrchk(cudaMalloc(&d_nnzPerVectorA, k * sizeof(*d_nnzPerVectorA)));
    cusparseSafeCall(cusparseSnnz(handle, CUSPARSE_DIRECTION_ROW, m, k, descrA, d_A_dense, lda, d_nnzPerVectorA, &nnzA));

    // --- Device side number of nonzero elements per row of matrix B
    int *d_nnzPerVectorB;   gpuErrchk(cudaMalloc(&d_nnzPerVectorB, n * sizeof(*d_nnzPerVectorB)));
    cusparseSafeCall(cusparseSnnz(handle, CUSPARSE_DIRECTION_ROW, k, n, descrB, d_B_dense, lda, d_nnzPerVectorB, &nnzB));

    // --- Host side number of nonzero elements per row of matrix A
    int *h_nnzPerVectorA = (int *)malloc(k * sizeof(*h_nnzPerVectorA));
    gpuErrchk(cudaMemcpy(h_nnzPerVectorA, d_nnzPerVectorA, k * sizeof(*h_nnzPerVectorA), cudaMemcpyDeviceToHost));

    // --- Host side number of nonzero elements per row of matrix B
    int *h_nnzPerVectorB = (int *)malloc(n * sizeof(*h_nnzPerVectorB));
    gpuErrchk(cudaMemcpy(h_nnzPerVectorB, d_nnzPerVectorB, n * sizeof(*h_nnzPerVectorB), cudaMemcpyDeviceToHost));

    // --- Device side sparse matrix
    float *d_A;            gpuErrchk(cudaMalloc(&d_A, nnzA * sizeof(*d_A)));
    float *d_B;            gpuErrchk(cudaMalloc(&d_B, nnzB * sizeof(*d_B)));

    int *d_A_RowIndices;    gpuErrchk(cudaMalloc(&d_A_RowIndices, (m + 1) * sizeof(*d_A_RowIndices)));
    int *d_B_RowIndices;    gpuErrchk(cudaMalloc(&d_B_RowIndices, (k + 1) * sizeof(*d_B_RowIndices)));
    int *d_C_RowIndices;    gpuErrchk(cudaMalloc(&d_C_RowIndices, (m + 1) * sizeof(*d_C_RowIndices)));
    int *d_A_ColIndices;    gpuErrchk(cudaMalloc(&d_A_ColIndices, nnzA * sizeof(*d_A_ColIndices)));
    int *d_B_ColIndices;    gpuErrchk(cudaMalloc(&d_B_ColIndices, nnzB * sizeof(*d_B_ColIndices)));

    cusparseSafeCall(cusparseSdense2csr(handle, m, k, descrA, d_A_dense, lda, d_nnzPerVectorA, d_A, d_A_RowIndices, d_A_ColIndices));
    cusparseSafeCall(cusparseSdense2csr(handle, k, n, descrB, d_B_dense, lda, d_nnzPerVectorB, d_B, d_B_RowIndices, d_B_ColIndices));

    // --- Host side sparse matrices
    float *h_A = (float *)malloc(nnzA * sizeof(*h_A));
    float *h_B = (float *)malloc(nnzB * sizeof(*h_B));
    int *h_A_RowIndices = (int *)malloc((m + 1) * sizeof(*h_A_RowIndices));
    int *h_A_ColIndices = (int *)malloc(nnzA * sizeof(*h_A_ColIndices));
    int *h_B_RowIndices = (int *)malloc((k + 1) * sizeof(*h_B_RowIndices));
    int *h_B_ColIndices = (int *)malloc(nnzB * sizeof(*h_B_ColIndices));
    int *h_C_RowIndices = (int *)malloc((m + 1) * sizeof(*h_C_RowIndices));
    gpuErrchk(cudaMemcpy(h_A, d_A, nnzA * sizeof(*h_A), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_A_RowIndices, d_A_RowIndices, (m + 1) * sizeof(*h_A_RowIndices), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_A_ColIndices, d_A_ColIndices, nnzA * sizeof(*h_A_ColIndices), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_B, d_B, nnzB * sizeof(*h_B), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_B_RowIndices, d_B_RowIndices, (k + 1) * sizeof(*h_B_RowIndices), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_B_ColIndices, d_B_ColIndices, nnzB * sizeof(*h_B_ColIndices), cudaMemcpyDeviceToHost));

    // --- Performing the matrix - matrix multiplication
    int baseC, nnzC = 0;
    // nnzTotalDevHostPtr points to host memory
    int *nnzTotalDevHostPtr = &nnzC;

    cusparseSafeCall(cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST));

    cusparseSafeCall(cusparseXcsrgemmNnz(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, k, descrB, nnzB,
                                         d_B_RowIndices, d_B_ColIndices, descrA, nnzA, d_A_RowIndices, d_A_ColIndices, descrC, d_C_RowIndices,
                                         nnzTotalDevHostPtr));
    if (NULL != nnzTotalDevHostPtr) nnzC = *nnzTotalDevHostPtr;
    else {
        gpuErrchk(cudaMemcpy(&nnzC,  d_C_RowIndices + m, sizeof(int), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(&baseC, d_C_RowIndices,     sizeof(int), cudaMemcpyDeviceToHost));
        nnzC -= baseC;
    }
    int *d_C_ColIndices;    gpuErrchk(cudaMalloc(&d_C_ColIndices, nnzC * sizeof(int)));
    float *d_C;            gpuErrchk(cudaMalloc(&d_C, nnzC * sizeof(float)));
    float *h_C = (float *)malloc(nnzC * sizeof(*h_C));
    int *h_C_ColIndices = (int *)malloc(nnzC * sizeof(*h_C_ColIndices));
    cusparseSafeCall(cusparseScsrgemm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, k, descrB, nnzB,
                                      d_B, d_B_RowIndices, d_B_ColIndices, descrA, nnzA, d_A, d_A_RowIndices, d_A_ColIndices, descrC,
                                      d_C, d_C_RowIndices, d_C_ColIndices));

    cusparseSafeCall(cusparseScsr2dense(handle, m, n, descrC, d_C, d_C_RowIndices, d_C_ColIndices, d_C_dense, lda));

    gpuErrchk(cudaMemcpy(h_C ,           d_C,            nnzC * sizeof(*h_C), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_C_RowIndices, d_C_RowIndices, (m + 1) * sizeof(*h_C_RowIndices), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_C_ColIndices, d_C_ColIndices, nnzC * sizeof(*h_C_ColIndices), cudaMemcpyDeviceToHost));

    gpuErrchk(cudaMemcpy(h_C_dense, d_C_dense, m * n * sizeof(float), cudaMemcpyDeviceToHost));

}