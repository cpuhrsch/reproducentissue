__version__ = '0.0.1+6ca2e65'
git_version = '6ca2e65c394af437ceff687827162a04ccbf1292'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
