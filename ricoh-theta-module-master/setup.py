#!/usr/bin/env python
# -*- coding: utf-8 -*-
# The MIT License (MIT)
#
# Copyright (c) 2016 Evripidis Gkanias, Theodoros Stouraitis
#                   <ev.gkanias@gmail.com>, <stoutheo@gmail.com>
#

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='ThetaCV',
    version='0.1',
    description='',
    long_description=readme,
    author='Gkanias Evripidis, Stouraitis Theodoros',
    author_email='ev.gkanias@gmail.com, stoutheo@gmail.com',
    url='https://github.com/InsectRobotics/ricoh-theta-module',
    license=license,
    classifiers=[
        # How mature is this project?
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Compute Vision :: Processing Tools',

        # License
        'License :: MIT License',

        # Specify the Python versions we support here.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
    ],
    keywords='computer vision development fiddler crab 360camera ricoh theta s',
    install_requires=['cython', 'setuptools', 'nose', 'numpy', 'scipy', 'matplotlib', 'argparse', 'pyyaml', 'imutils'],
    packages=find_packages(exclude=('examples', 'repos-3rd', 'venv')),
    package_data={
        'thetacv': [
            'logs/error.log',
            'logs/info.log',
            'subsampling/data/smolka.csv',
            'subsampling/data/custom.csv',
            'subsampling/params.yaml',
            'capture/params.yaml',
            'capture/logging.yaml',
            'capture/MAP/blur.ker',
            'capture/MAP/voronoi_custom.map',
            'capture/MAP/voronoi_smolka.map',
        ],
    }
)
