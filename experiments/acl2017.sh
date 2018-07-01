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
mkdir -p hershcovich2017a
cd hershcovich2017a
pip install --user virtualenv
python -m virtualenv --python=/usr/bin/python3 venv
. venv/bin/activate              # on bash
pip install "tupa>=1.0,<1.1"
git clone https://github.com/huji-nlp/ucca-corpora --branch v1.2
mkdir -p models wiki-sentences 20k-sentences
python -m scripts.standard_to_sentences ucca-corpora/wiki/xml -o wiki-sentences
python -m scripts.standard_to_sentences ucca-corpora/vmlslm/en -o 20k-sentences
python -m scripts.split_corpus wiki-sentences -t 4268 -d 454 -l
curl -L --remote-name-all https://github.com/huji-nlp/tupa/releases/download/v1.0/{sparse,mlp,bilstm}.tgz
tar xvzf sparse.tgz
tar xvzf mlp.tgz
tar xvzf bilstm.tgz
python -m spacy download en_core_web_lg
for TEST_SET in wiki-sentences/test 20k-sentences; do
    for MODEL in sparse mlp bilstm; do
        python -m tupa.parse -c $MODEL -m models/ucca-$MODEL -We $TEST_SET
    done
done
