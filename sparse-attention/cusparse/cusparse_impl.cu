#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>

#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include "../utils/cycleTimer.h"


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

void print_a_row(float *dense, int row, int width) {
    printf("row = %d, width = %d\n", row, width);
    for (int i = 0; i < width; i++) {
        printf("%f ", dense[row * width + i]);
    }
    printf("\n");
}

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

    // --- Descriptors for sparse matrices
    cusparseMatDescr_t descrA, descrB, descrC;
    cusparseSafeCall(cusparseCreateMatDescr(&descrA));
    cusparseSafeCall(cusparseCreateMatDescr(&descrB));
    cusparseSafeCall(cusparseCreateMatDescr(&descrC));

    int nnzA = 0;                           // --- Number of nonzero elements in dense matrix A
    int nnzB = 0;                           // --- Number of nonzero elements in dense matrix B

    const int lda = m;                      // --- Leading dimension of dense matrix
    const int ldb = k;                      // --- Leading dimension of dense matrix

    // --- Device side number of nonzero elements per row
    // TODO: is the size of d_nnzPerVectorA correct?
    int *d_nnzPerVectorA, *d_nnzPerVectorB;
    gpuErrchk(cudaMalloc(&d_nnzPerVectorA, m * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_nnzPerVectorB, k * sizeof(int)));
    cusparseSafeCall(cusparseSnnz(handle, CUSPARSE_DIRECTION_ROW, m, k, descrA, d_A_dense, lda, d_nnzPerVectorA, &nnzA));
    cusparseSafeCall(cusparseSnnz(handle, CUSPARSE_DIRECTION_ROW, k, n, descrB, d_B_dense, ldb, d_nnzPerVectorB, &nnzB));

    // --- Host side number of nonzero elements per row
    int *h_nnzPerVectorA = (int *)malloc(m * sizeof(*h_nnzPerVectorA));
    int *h_nnzPerVectorB = (int *)malloc(k * sizeof(*h_nnzPerVectorB));
    gpuErrchk(cudaMemcpy(h_nnzPerVectorA, d_nnzPerVectorA, k * sizeof(*h_nnzPerVectorA), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_nnzPerVectorB, d_nnzPerVectorB, n * sizeof(*h_nnzPerVectorB), cudaMemcpyDeviceToHost));

    // --- Device side sparse matrix
    float *d_A, *d_B;
    gpuErrchk(cudaMalloc(&d_A, nnzA * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_B, nnzB * sizeof(float)));

    int *d_A_RowIndices;    gpuErrchk(cudaMalloc(&d_A_RowIndices, (m + 1) * sizeof(int)));
    int *d_B_RowIndices;    gpuErrchk(cudaMalloc(&d_B_RowIndices, (k + 1) * sizeof(int)));
    int *d_C_RowIndices;    gpuErrchk(cudaMalloc(&d_C_RowIndices, (m + 1) * sizeof(int)));
    int *d_A_ColIndices;    gpuErrchk(cudaMalloc(&d_A_ColIndices, nnzA * sizeof(int)));
    int *d_B_ColIndices;    gpuErrchk(cudaMalloc(&d_B_ColIndices, nnzB * sizeof(int)));

    cusparseSafeCall(cusparseSdense2csr(handle, m, k, descrA, d_A_dense, lda, d_nnzPerVectorA, d_A, d_A_RowIndices, d_A_ColIndices));
    cusparseSafeCall(cusparseSdense2csr(handle, k, n, descrB, d_B_dense, ldb, d_nnzPerVectorB, d_B, d_B_RowIndices, d_B_ColIndices));

    // --- Performing the matrix - matrix multiplication
    int baseC, nnzC = 0;
    // nnzTotalDevHostPtr points to host memory
    int *nnzTotalDevHostPtr = &nnzC;

    cusparseSafeCall(cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST));

    cusparseSafeCall(cusparseXcsrgemmNnz(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         m, n, k, descrA, nnzA, d_A_RowIndices, d_A_ColIndices, descrB, nnzB,
                                         d_B_RowIndices, d_B_ColIndices, descrC, d_C_RowIndices, nnzTotalDevHostPtr));
    if (NULL != nnzTotalDevHostPtr) nnzC = *nnzTotalDevHostPtr;
    else {
        gpuErrchk(cudaMemcpy(&nnzC,  d_C_RowIndices + m, sizeof(int), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(&baseC, d_C_RowIndices,     sizeof(int), cudaMemcpyDeviceToHost));
        nnzC -= baseC;
    }
    int *d_C_ColIndices;    gpuErrchk(cudaMalloc(&d_C_ColIndices, nnzC * sizeof(int)));
    float *d_C;            gpuErrchk(cudaMalloc(&d_C, nnzC * sizeof(float)));

    double start = CycleTimer::currentSeconds();
    cusparseScsrgemm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, k,
                     descrA, nnzA, d_A, d_A_RowIndices, d_A_ColIndices,
                     descrB, nnzB, d_B, d_B_RowIndices, d_B_ColIndices,
                     descrC, d_C, d_C_RowIndices, d_C_ColIndices);
    double end = CycleTimer::currentSeconds();
    printf("cusparse matmul:    %.4f ms\n", 1000.f * (end - start));

    cusparseSafeCall(cusparseScsr2dense(handle, m, n, descrC, d_C, d_C_RowIndices, d_C_ColIndices, d_C_dense, lda));

    gpuErrchk(cudaMemcpy(h_C_dense, d_C_dense, m * n * sizeof(float), cudaMemcpyDeviceToHost));
    // cusparse return col-major result, while we implemented row-splitting to return row-major results
    // this is unimportant regarding performance, since we can trivially change row-splitting to return in col-major.
    print_a_row(h_C_dense, 0, n);

}