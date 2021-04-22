#pragma once
#include <nestedtensor/csrc/nested_tensor_impl.h>

namespace torch {
namespace nested_tensor {

at::Tensor nested_tensor_impl(int64_t);

} // namespace nested_tensor
} // namespace torch
