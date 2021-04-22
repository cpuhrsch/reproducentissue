#include <ATen/ATen.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/op_registration/op_registration.h>
#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/library.h>

namespace at {

using namespace c10;

NestedTensorImpl::NestedTensorImpl(int64_t payload)
    : TensorImpl(c10::DispatchKeySet({NestedTensorKey}), at::ones({}).dtype(),
                 at::ones({}).device()) {
  remove_autograd_key();
  key_set_ = key_set_ - c10::DispatchKeySet({DispatchKey::InplaceOrView});
  payload_ = payload;
}

} // namespace at
