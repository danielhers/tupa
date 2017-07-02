#!/usr/bin/env bash

# Make sure we have all submodules
git submodule update --init --recursive

# Install Python requirements
pip install -r requirements.txt
python -m spacy download en
python -m nltk.downloader propbank
rm -rf ~/nltk_data/corpora/propbank
git clone https://github.com/propbank/propbank-frames ~/nltk_data/corpora/propbank
pip install git+https://github.com/clab/dynet#egg=dynet

# Download AMR resources
cd "$(dirname "$0")"/scheme/util/resources
curl --remote-name-all https://amr.isi.edu/download/lists/{{have-{org,rel}-role-91-roles,verbalization-list}-v1.06,morph-verbalization-v1.01}.txt
