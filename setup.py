#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

def calculate_version():
    initpy = open('skrebate/_version.py').read().split('\n')
    version = list(filter(lambda x: '__version__' in x, initpy))[0].split('\'')[1]
    return version

package_version = calculate_version()

setup(
    name='skrebate',
    version=package_version,
    author='Randal S. Olson, Pete Schmitt, and Ryan J. Urbanowicz',
    author_email='rso@randalolson.com, ryanurb@upenn.edu',
    packages=find_packages(),
    url='https://github.com/EpistasisLab/scikit-rebate',
    license='License :: OSI Approved :: MIT License',
    description=('Relief-based feature selection algorithms'),
    long_description='''
A sklearn-compatible Python implementation of ReBATE, a suite of Relief-based feature selection algorithms.

Contact
=============
If you have any questions or comments about skrebate, please feel free to contact us via e-mail: rso@randalolson.com and ryanurb@upenn.edu

This project is hosted at https://github.com/EpistasisLab/scikit-rebate
''',
    zip_safe=True,
    install_requires=['numpy', 'scipy', 'scikit-learn'],
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Utilities'
    ],
    keywords=['data mining', 'feature selection', 'feature importance', 'machine learning', 'data analysis', 'data engineering', 'data science'],
    include_package_data=True,
)
