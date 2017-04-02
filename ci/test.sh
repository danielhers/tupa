#!/usr/bin/env bash

case "$TEST_SUITE" in
unit)
    # unit tests
    python -m unittest discover -v || exit 1
    # basic parser tests
    for m in "" s a; do
      python tupa/parse.py -I 10 -t ucca/doc/toy.xml -d ucca/doc/toy.xml -"$m"m model_toy$m -v || exit 1
      python tupa/parse.py ucca/doc/toy.xml -e"$m"m model_toy$m -v || exit 1
    done
    python tupa/parse.py -f amr -I 10 -t test_files/LDC2014T12.txt -d test_files/LDC2014T12.txt -sm model_LDC2014T12 -v || exit 1
    python tupa/parse.py -f amr test_files/LDC2014T12.txt -esm model_LDC2014T12 -v || exit 1
    ;;
sparse)
    python tupa/parse.py -v -c sparse --max-words-external=5000 -Web pickle/dev/*0.pickle -t pickle/train/*0.pickle
    ;;
mlp)
    python tupa/parse.py -v -c mlp --max-words-external=5000 --layer-dim=100 -Web pickle/dev/*0.pickle -t pickle/train/*0.pickle --dynet-mem=1500
    ;;
bilstm)
    python tupa/parse.py -v -c bilstm --max-words-external=5000 --layer-dim=100 -Web pickle/dev/*0.pickle -t pickle/train/*0.pickle --dynet-mem=1500
    ;;
tune)
    export PARAMS_NUM=5
    while :; do
      python tupa/tune.py ucca/doc/toy.xml -t ucca/doc/toy.xml --max-words-external=5000 --dynet-mem=1500 && break
    done
    column -t -s, params.csv
    ;;
convert)
    python contrib/convert_and_evaluate.py alignment-release-*-bio.txt
    ;;
esac
