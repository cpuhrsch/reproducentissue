#pragma once
#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <c10/util/Metaprogramming.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/extension.h>
#include <torch/library.h>

namespace at {

constexpr auto NestedTensorKey = DispatchKey::NestedTensor;

struct NestedTensorImpl;

template <class A>
bool is_nested_tensor_impl(A tensor) {
  return tensor.unsafeGetTensorImpl()->key_set().has(at::NestedTensorKey);
}

template <class A, class B>
bool is_nested_tensor_impl(A first, B other) {
  return is_nested_tensor_impl(first) && is_nested_tensor_impl(other);
}

template <class A, class B, class... C>
bool is_nested_tensor_impl(A first, B second, C... other) {
  return is_nested_tensor_impl(first, second) &&
      is_nested_tensor_impl(other...);
}

struct NestedTensorImpl : public c10::TensorImpl {
  explicit NestedTensorImpl(bool none);

  int64_t dim() const override {
    return 0;
  }
  int64_t numel() const override {
    return 0;
  }
  bool is_contiguous(at::MemoryFormat memory_format) const override {
    return true;
  }
  IntArrayRef sizes() const override {
    TORCH_CHECK(
        false,
        "Internal error: NestedTensorImpl doesn't support sizes. Please file an issue on https://github.com/pytorch/nestedtensor");
    return IntArrayRef(_sizes);
  }
  IntArrayRef strides() const override;

 private:
  std::vector<int64_t> _sizes;
};


template <class FuncPtr, class ParameterTypes>
struct _Function_trace_wrapper {};

template <class FuncPtr, class... Parameters>
struct _Function_trace_wrapper<
    FuncPtr,
    c10::guts::typelist::typelist<Parameters...>> {
  using ReturnType = typename c10::guts::infer_function_traits_t<
      typename FuncPtr::FuncType>::return_type;
  static ReturnType apply(Parameters... args) {
    std::cout << "Calling " << typeid(FuncPtr).name() << std::endl;
    return (*FuncPtr::func_ptr())(args...);
  }
};

template <class FuncPtr>
constexpr auto trace(FuncPtr /*func_ptr*/) {
  using function_traits =
      c10::guts::infer_function_traits_t<typename FuncPtr::FuncType>;
  using parameter_types = typename function_traits::parameter_types;
  return &_Function_trace_wrapper<FuncPtr, parameter_types>::apply;
}

#ifdef TRACEPACKED
// #define nt_impl(M, NAME, FUNC) M.impl_UNBOXED(NAME, trace(TORCH_FN(FUNC)))
#define nt_impl(M, NAME, FUNC) \
  M.impl(                      \
      NAME,                    \
      torch::CppFunction::makeFromUnboxedFunction(trace(TORCH_FN(FUNC))))
#else
// #define nt_impl(M, NAME, FUNC) M.impl_UNBOXED(NAME, FUNC)
#define nt_impl(M, NAME, FUNC) M.impl(NAME, TORCH_FN(FUNC))
#endif

} // namespace at
