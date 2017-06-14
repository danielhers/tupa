#!/usr/bin/env bash

# Make sure we have all submodules
git submodule update --init --recursive

# Install Python requirements
pip install -r requirements.txt
python -m spacy download en > spacy.log
python -m nltk.downloader propbank
rm -rf ~/nltk_data/corpora/propbank
git clone https://github.com/propbank/propbank-frames ~/nltk_data/corpora/propbank
pip install git+https://github.com/clab/dynet#egg=dynet
