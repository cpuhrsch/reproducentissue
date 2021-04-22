import torch
import nestedtensor
a = nestedtensor._C.nested_tensor_impl(4)
b = torch.matmul(a, torch.tensor([1]))
