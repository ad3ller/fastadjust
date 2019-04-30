# -*- coding: utf-8 -*-
import sys
from setuptools import setup

setup(name='fastadjust',
      version='0.0.4',
      description='fast adjust SIMION potential arrays',
      url='',
      author='Adam Deller',
      author_email='a.deller@ucl.ac.uk',
      license='BSD',
      packages=['fastadjust'],
      install_requires=[
          'scipy>=0.14','numpy>=1.10','h5py>=2.7.0'
      ],
      include_package_data=False,
      zip_safe=False)
