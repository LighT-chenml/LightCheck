# from distutils.core import setup, Extension
# from setuptools import setup, Extension
from setuptools import setup
from torch.utils import cpp_extension

module1 = cpp_extension.CUDAExtension('pmemop', libraries=['pmem'], sources=['pmemop_module.cpp', 'pmemop_object.cpp', 'pmemop_alloc.cpp', 'ready_event.cpp'],
      include_dirs=['/usr/local/lib/python3.6/site-packages/torch/include/', '/usr/local/lib/python3.6/site-packages/torch/include/torch/csrc/api/include/']
      )
      # pyrex_gdb=True, extra_compile_args=["-g"], extra_link_args=["-g"])

setup(name='pmemop',
      version='1.0',
      description='This is a pmemop package',
     #  cmdclass = {'build_ext': build_ext},
      cmdclass={'build_ext': cpp_extension.BuildExtension},
      ext_modules=[module1])
