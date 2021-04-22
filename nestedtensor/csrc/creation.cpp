#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/extension.h>

namespace py = pybind11;

namespace torch {
namespace nested_tensor {

using namespace torch::jit;

at::Tensor nested_tensor_impl(
    py::sequence list,
    py::object dtype_,
    py::object device_,
    bool requires_grad,
    bool pin_memory) {
  auto result = at::detail::make_tensor<NestedTensorImpl>(false);
  return result;
}

} // namespace nested_tensor
} // namespace torch
