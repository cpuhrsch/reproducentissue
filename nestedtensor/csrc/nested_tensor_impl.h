#pragma once
#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <c10/util/Metaprogramming.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/extension.h>
#include <torch/library.h>

namespace at {

constexpr auto NestedTensorKey = DispatchKey::NestedTensor;

struct NestedTensorImpl : public c10::TensorImpl {
  explicit NestedTensorImpl(int64_t payload);
  IntArrayRef sizes() const override {
    TORCH_CHECK(
        false,
        "Internal error: NestedTensorImpl doesn't support sizes. Please file an issue on https://github.com/pytorch/nestedtensor");
    return IntArrayRef(_sizes);
  }

private:
  int64_t payload_;
  std::vector<int64_t> _sizes;
};

#define nt_impl(M, NAME, FUNC) M.impl(NAME, TORCH_FN(FUNC))

} // namespace at
