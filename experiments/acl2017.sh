#!/usr/bin/env bash
set -e
# Reproduce experiments from the paper:
# @InProceedings{hershcovich2017a,
#   author    = {Hershcovich, Daniel  and  Abend, Omri  and  Rappoport, Ari},
#   title     = {A Transition-Based Directed Acyclic Graph Parser for UCCA},
#   booktitle = {Proc. of ACL},
#   year      = {2017},
#   pages     = {1127--1138},
#   url       = {http://aclweb.org/anthology/P17-1104}
# }
#
# On Ubuntu, requires first installing the following packages:
# python3-distutils python3-dev cmake libboost-all-dev
mkdir -p hershcovich2017a
cd hershcovich2017a
if python -m virtualenv --version || pip install --user virtualenv; then
    python -m virtualenv --python=/usr/bin/python3 venv
    . venv/bin/activate              # on bash
fi
pip install cython
pip install "tupa>=1.0,<1.1"
git clone https://github.com/huji-nlp/ucca-corpora --branch v1.2
mkdir -p models wiki-sentences 20k-sentences
if [[ -z "$CORPUS" || "$CORPUS" == wiki ]]; then
    python -m scripts.standard_to_sentences ucca-corpora/wiki/xml/*.xml -o wiki-sentences
    python -m scripts.split_corpus wiki-sentences -t 4268 -d 454 -l
fi
if [[ -z "$CORPUS" || "$CORPUS" == 20k ]]; then
    python -m scripts.standard_to_sentences ucca-corpora/vmlslm/en/*.xml -o 20k-sentences
fi
curl -L --remote-name-all https://github.com/huji-nlp/tupa/releases/download/v1.0/{sparse,mlp,bilstm}.tgz
tar xvzf sparse.tgz
tar xvzf mlp.tgz
tar xvzf bilstm.tgz
python -m spacy download en
for TEST_SET in wiki-sentences/test 20k-sentences; do
    if [[ -z "$CORPUS" || $TEST_SET == "$CORPUS"* ]]; then
        for MODEL in sparse mlp bilstm; do
            python -m tupa.parse -c $MODEL -m models/$MODEL -We $TEST_SET
        done
    fi
done
