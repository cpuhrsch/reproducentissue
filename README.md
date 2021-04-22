### Build

```
clear && python setup.py clean && DEBUG=0 USE_NINJA=1 pip install -v -e .
```

### Reproduce error

Run
```
python repro.py
```

which contains

```
import torch
import nestedtensor
a = nestedtensor._C.nested_tensor_impl(4)
b = torch.matmul(a, torch.tensor([1]))
```

Expecting this to print and then immediately exit from within the matmul kernel registered in nestedtensor/csrc/matmul.cpp.
