#!/usr/bin/env python
from setuptools import setup, find_packages
setup(name='NequickG',
      version='1.0.0',
      description='NequickG (fork from Fraunhofer IIS)',
      url='https://github.com/Fraunhofer-IIS/NequickG',
      keyword=['NequickG'],
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
	  install_requires=['numpy', 'matplotlib'])

