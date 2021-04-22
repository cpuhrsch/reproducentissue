__version__ = '0.0.1+f630f76'
git_version = 'f630f76f58484695a9cb8f6c35aaff85869b6afb'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
