__version__ = '0.0.1'
git_version = 'Unknown'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
