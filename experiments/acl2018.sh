#!/usr/bin/env bash
set -e
# Reproduce experiments from the paper:
# @InProceedings{hershcovich2018multitask,
#   author    = {Hershcovich, Daniel  and  Abend, Omri  and  Rappoport, Ari},
#   title     = {Multitask Parsing Across Semantic Representations},
#   booktitle = {Proc. of ACL},
#   year      = {2018},
#   url       = {http://www.cs.huji.ac.il/~danielh/acl2018.pdf}
# }
mkdir -p hershcovich2018multitask
cd hershcovich2018multitask
pip install --user virtualenv
python -m virtualenv --python=/usr/bin/python3 venv
. venv/bin/activate              # on bash
pip install "tupa==1.3.2"
git clone https://github.com/huji-nlp/ucca-corpora --branch v1.2
mkdir -p models wiki-sentences 20k{,-fr,-de}-sentences
python -m scripts.standard_to_sentences ucca-corpora/wiki/xml -o wiki-sentences
python -m scripts.standard_to_sentences ucca-corpora/vmlslm/en -o 20k-sentences
python -m scripts.standard_to_sentences ucca-corpora/vmlslm/fr -o 20k-fr-sentences
python -m scripts.standard_to_sentences ucca-corpora/20k_de/xml -o 20k-de-sentences
python -m scripts.split_corpus wiki-sentences -t 4268 -d 454 -l
python -m scripts.split_corpus 20k-fr-sentences -t 413 -d 67 -l
python -m scripts.split_corpus 20k-de-sentences -t 3429 -d 561 -l
curl -L --remote-name-all http://www.cs.huji.ac.il/~danielh/ucca/ucca{,-amr,-dm,-ud++,-amr-dm,-amr-ud++,-amr-dm-ud++}-bilstm-1.3.2{,-fr,-de}.tar.gz
tar xvzf ucca-*.tar.gz
for TEST_SET in wiki-sentences/test 20k-sentences; do
    for AUX in "" -amr -dm -ud++ -amr-dm -amr-ud++ -amr-dm-ud++; do
        python -m tupa -m models/ucca$AUX-bilstm -We $TEST_SET
    done
done
for LANG in fr de; do
    python -m tupa -m models/ucca-bilstm-$LANG -We 20k-$LANG-sentences/test --lang $LANG
    python -m tupa -m models/ucca-ud-bilstm-$LANG -We 20k-$LANG-sentences/test --lang $LANG
done
