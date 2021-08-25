from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='kmeanpp',
      ext_modules=[
          cpp_extension.CppExtension('kmeanpp', ['kmeanpp.cc'], 
          extra_compile_args=['-fopenmp', '-std=c++17'])
      ],
      cmdclass={'build_ext': cpp_extension.BuildExtension},
      )