__version__ = '0.0.1+907b60f'
git_version = '907b60f32405187eb6c67e03f83e44e9f77ae8e2'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
