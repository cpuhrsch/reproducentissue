#include <nestedtensor/csrc/creation.h>
#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <torch/csrc/Size.h>
#include <torch/csrc/autograd/python_variable_indexing.h>
#include <torch/extension.h>


namespace py = pybind11;

using namespace torch::nested_tensor;
using namespace at;


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  m.def("nested_tensor_impl", &torch::nested_tensor::nested_tensor_impl);

}
