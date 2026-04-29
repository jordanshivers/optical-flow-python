from setuptools import setup, find_packages

setup(
    name='optical_flow',
    version='1.1.0',
    description='Optical flow estimation: Python reimplementation of Sun, Roth & Black (CVPR 2010)',
    author='Original: Deqing Sun, Stefan Roth, Michael J. Black; Python port',
    license='Research Use Only - See LICENSE file',
    packages=find_packages(include=['optical_flow', 'optical_flow.*',
                                    'flow_fast', 'flow_fast.*']),
    python_requires='>=3.9',
    install_requires=[
        'numpy>=1.21',
        'scipy>=1.7',
        'matplotlib>=3.4',
        'Pillow>=8.0',
        'scikit-image>=0.19',
    ],
    extras_require={
        'fast': [
            'numba>=0.58',
            'opencv-python>=4.5',
        ],
        'cholmod': [
            'scikit-sparse>=0.4.8',
        ],
        'deep': [
            'torch>=1.6',
            'torchvision>=0.10',
            'timm',
            'safetensors',
            'huggingface_hub',
            'gdown',
        ],
        'dev': [
            'pytest>=7.0',
            'jupyter',
        ],
    },
    package_data={
        '': ['data/**/*'],
    },
    include_package_data=True,
    test_suite='tests',
    tests_require=['pytest>=7.0'],
)
