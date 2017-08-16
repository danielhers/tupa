#!/usr/bin/env bash

case "$TEST_SUITE" in
unit)
    # unit tests
    python -m unittest discover -v || exit 1
    # basic parser tests
    for m in "" --sentences --paragraphs; do
      python tupa/parse.py -I 10 -t test_files/toy.xml -d test_files/toy.xml "$m" -m model_toy$m -v || exit 1
      python tupa/parse.py test_files/toy.xml "$m" -em model_toy$m -v || exit 1
    done
    python tupa/parse.py -f amr -I 10 -t test_files/LDC2014T12.txt -d test_files/LDC2014T12.txt -m model_LDC2014T12 -v || exit 1
    python tupa/parse.py -f amr test_files/LDC2014T12.txt -em model_LDC2014T12 -v || exit 1
    ;;
sparse-ucca)
    python tupa/parse.py -v -c sparse -We pickle/dev/*0.pickle -t pickle/train/*0.pickle
    ;;
mlp-ucca)
    python tupa/parse.py -v -c mlp --max-words-external=5000 --layer-dim=100 -We pickle/dev/*0.pickle -t pickle/train/*0.pickle
    ;;
bilstm-ucca)
    python tupa/parse.py -v -c bilstm --max-words-external=5000 --layer-dim=100 -We pickle/dev/*0.pickle -t pickle/train/*0.pickle
    ;;
noop-ucca)
    python tupa/parse.py -v -c noop -We pickle/dev/*0.pickle -t pickle/train/*0.pickle
    ;;
tune-ucca)
    export PARAMS_NUM=5
    while :; do
      python tupa/tune.py -t test_files/toy.xml -d test_files/toy.xml --max-words-external=1000 --max-node-labels=250 && break
    done
    column -t -s, params.csv
    ;;
sparse-amr)
    python tupa/parse.py -v -c sparse --max-node-labels=250 -We -f amr alignment-release-dev-bio.txt -t alignment-release-training-bio/*10.txt --no-wikification
    ;;
noop-amr)
    python tupa/parse.py -v -c noop -We -f amr alignment-release-dev-bio.txt -t alignment-release-training-bio --no-wikification
    ;;
convert-amr)
    python scheme/convert_and_evaluate.py alignment-release-dev-bio.txt -v
    ;;
esac
