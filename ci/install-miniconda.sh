#!/usr/bin/env bash

if [ -d $HOME/miniconda ]; then
    echo $HOME/miniconda already exists
else
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
    conda install --yes python=$TRAVIS_PYTHON_VERSION pip numpy scipy libgcc
fi