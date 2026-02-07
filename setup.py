from setuptools import setup, find_packages

setup(
    name='optical_flow',
    version='1.0.0',
    description='Optical flow estimation: Python reimplementation of Sun, Roth & Black (CVPR 2010)',
    author='Original: Deqing Sun, Stefan Roth, Michael J. Black; Python port',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.21',
        'scipy>=1.7',
        'matplotlib>=3.4',
        'Pillow>=8.0',
        'scikit-image>=0.19',
    ],
    extras_require={
        'dev': ['pytest>=7.0', 'jupyter'],
    },
    package_data={
        '': ['data/**/*'],
    },
    include_package_data=True,
    test_suite='tests',
    tests_require=['pytest>=7.0'],
)
