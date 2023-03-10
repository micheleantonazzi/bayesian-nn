import os
import re

# To use a consistent encoding
from codecs import open as copen

from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the relevant file
with copen(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


def read(*parts):
    with copen(os.path.join(here, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


__version__ = find_version("bayesian_nn", "__version__.py")

test_deps = [
    "pytest",
    "pytest-cov",
]

extras = {
    'test': test_deps,
}

setup(
    name='bayesian-nn',
    version=__version__,
    description="Bayesian neural network",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/micheleantonazzi/bayesian-nn",
    author="Michele Antonazzi",
    author_email="micheleantonazzi@gmail.com",
    # Choose your license
    license='MIT',
    include_package_data=True,
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3'
    ],
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    tests_require=test_deps,
    # Add here the package dependencies
    install_requires=[
        'torch',
        'torchvision',
        'pyro-ppl',
        'hamiltorch @ git+https://github.com/AdamCobb/hamiltorch',
        'matplotlib'
    ],
    entry_points={
        'console_scripts': [
        ],
    },
    test_deps=test_deps,
    extras_require=extras,
)