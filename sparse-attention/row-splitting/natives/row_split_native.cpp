#include <torch/extension.h>
#include <vector>
#include <cuda.h>

void row_split_spmm(torch::Tensor *h_A_dense, torch::Tensor *h_B_dense, torch::Tensor *h_C_dense, int m, int k, int n);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("row_split_spmm", &row_split_spmm, "row_split_spmm");
}
