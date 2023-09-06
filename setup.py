import os
from setuptools import setup, find_packages

import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension



def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


def get_requirements(filename='requirements.txt'):
    here = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(here, filename), 'r') as f:
        requires = [line.replace('\n', '') for line in f.readlines()]
    return requires


def make_cuda_ext(name, module, sources):
    define_macros = []

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
    else:
        raise EnvironmentError('CUDA is required to compile!')

    return CUDAExtension(
        name='{}.{}'.format(module, name),
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        define_macros=define_macros,
        extra_compile_args={
            'cxx': [],
            'nvcc': [
                '-D__CUDA_NO_HALF_OPERATORS__',
                '-D__CUDA_NO_HALF_CONVERSIONS__',
                '-D__CUDA_NO_HALF2_OPERATORS__',
            ]
        })


if __name__ == '__main__':
    setup(
        name='detection-automotive',
        version='1.3',
        description='Automotive Object Detection',
        long_description=readme(),
        long_description_content_type='text/markdown',
        url='',
        author='',
        author_email='',
        classifiers=[],
        keywords='object detection',
        packages=find_packages(include=["detection_vis_backend.networks.*"]),
        package_data={'detection_vis_backend.networks.ops': ['*/*.so']},
        python_requires='>=3.8',
        install_requires=[],
        ext_modules=[
            # make_cuda_ext(
            #     name='deform_conv_2d_cuda',
            #     module='rodnet.ops.dcn',
            #     sources=[
            #         'src/deform_conv_2d_cuda.cpp',
            #         'src/deform_conv_2d_cuda_kernel.cu'
            #     ]),
            make_cuda_ext(
                name='deform_conv_3d_cuda',
                module='detection_vis_backend.networks.ops.dcn',
                sources=[
                    'src/deform_conv_3d_cuda.cpp',
                    'src/deform_conv_3d_cuda_kernel.cu'
                ]),
            # make_cuda_ext(
            #     name='deform_pool_2d_cuda',
            #     module='rodnet.ops.dcn',
            #     sources=[
            #         'src/deform_pool_2d_cuda.cpp',
            #         'src/deform_pool_2d_cuda_kernel.cu'
            #     ]),
            # make_cuda_ext(
            #     name='deform_pool_3d_cuda',
            #     module='rodnet.ops.dcn',
            #     sources=[
            #         'src/deform_pool_3d_cuda.cpp',
            #         'src/deform_pool_3d_cuda_kernel.cu'
            #     ]),
        ],
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False
    )
