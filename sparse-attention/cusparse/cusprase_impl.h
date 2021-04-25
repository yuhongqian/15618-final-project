#ifndef __CUSPARSE_IMPL_H__
#define __CUSPARSE_IMPL_H__

void cusparse_mmul(const float *h_A_dense, const float *h_B_dense, int m, int k, int n);

#endif