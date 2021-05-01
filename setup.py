from distutils.core import setup
from setuptools import find_packages

setup(
    name='borel',
    version='0.3.0dev',
    packages=find_packages(),
    long_description=open('README.md').read(),
)
