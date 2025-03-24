from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch
from pybind11 import get_include

ext_modules = [
    CUDAExtension(
		name='dtw_kernel_v4',
        sources=['dtw_kernel_v4.cu'],
        extra_compile_args={
            'nvcc': ['-O3', '-std=c++17', '-use_fast_math', '-Xcompiler', '-fPIC']
        },
        include_dirs=[
            get_include(),
            get_include(user=True),
            torch.utils.cpp_extension.include_paths(cuda=True)
        ],
    )
]

setup(
    name='dtw_kernel_v4',
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension
    }
)

# cp dtw_kernel_v4.cpp dtw_kernel_v4.cu && python -m pip uninstall dtw-kernel-v4 -y && TORCH_CUDA_ARCH_LIST="7.0+PTX" python dtw_kernel_v4_setup.py build_ext --inplace && TORCH_CUDA_ARCH_LIST="7.0+PTX" python dtw_kernel_v4_setup.py install
