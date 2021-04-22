#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <torch/extension.h>
#include <torch/library.h>

using namespace torch::nn;
namespace F = torch::nn::functional;

namespace at {

Tensor NestedTensor_matmul(const Tensor& self, const Tensor& other) {
  std::cout << "Calling NestedTensor_matmul" << std::endl;
  exit(1);
  return self;
}

Tensor& NestedTensor_matmul_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& result) {
  std::cout << "Calling NestedTensor_matmul_out" << std::endl;
  exit(1);
  return result;
}

TORCH_LIBRARY_IMPL(aten, NestedTensor, m) {
  nt_impl(m, "matmul", NestedTensor_matmul);
  nt_impl(m, "matmul.out", NestedTensor_matmul_out);
}
} // namespace at
