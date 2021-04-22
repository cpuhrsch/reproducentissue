#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/extension.h>

namespace py = pybind11;

namespace torch {
namespace nested_tensor {

using namespace torch::jit;

at::Tensor nested_tensor_impl(int64_t payload) {
  auto result = at::detail::make_tensor<NestedTensorImpl>(payload);
  return result;
}

} // namespace nested_tensor
} // namespace torch
