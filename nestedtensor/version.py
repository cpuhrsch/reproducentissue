__version__ = '0.0.1+5c3f032'
git_version = '5c3f032b658143ab75a4b804a02b38dfbdb0176d'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
