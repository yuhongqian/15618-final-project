#ifndef __UTILS_H__
#define __UTILS_H__

void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true);
void __cusparseSafeCall(cusparseStatus_t err, const char *file, const int line);

#endif