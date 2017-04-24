#!/usr/bin/env bash

# Make sure we have all submodules
git submodule update --init --recursive

# Install Python requirements
pip install -r requirements.txt
python -m spacy download en > spacy.log
python -m nltk.downloader wordnet propbank
git clone https://github.com/propbank/propbank-frames ~/nltk_data/corpora/propbank

# Install DyNet
export CXX="g++-4.8" CC="gcc-4.8"
git clone https://github.com/clab/dynet
cd dynet
hg clone https://bitbucket.org/eigen/eigen
mkdir build
ln -s eigen build/
cd build
cmake .. -DEIGEN3_INCLUDE_DIR=eigen -DPYTHON=`which python`
make
cd python
python setup.py install

# Install UCCA
cd ucca
python setup.py install
