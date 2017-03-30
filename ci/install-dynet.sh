#!/usr/bin/env bash

export CXX="g++-4.8" CC="gcc-4.8"
git clone https://github.com/clab/dynet
cd dynet
mkdir build
cd build
hg clone https://bitbucket.org/eigen/eigen
cmake .. -DEIGEN3_INCLUDE_DIR=eigen -DPYTHON=`which python`
make
cd python
python setup.py install
