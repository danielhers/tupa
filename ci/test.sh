#!/usr/bin/env bash

ACTION=${TEST_SUITE%-*}
FORMAT=${TEST_SUITE#*-}

# download data
case "$FORMAT" in
ucca)
    mkdir pickle
    curl -L http://www.cs.huji.ac.il/~danielh/ucca/ucca_corpus_pickle.tgz | tar xz -C pickle || curl -L https://www.dropbox.com/s/q4ycn45zlmhuf9k/ucca_corpus_pickle.tgz | tar xz -C pickle
    python -m scripts.split_corpus -q pickle -t 4282 -d 454 -l
    ;;
amr)
    curl --remote-name-all https://amr.isi.edu/download/2016-03-14/alignment-release-{training,dev,test}-bio.txt
    python scheme/split.py -q alignment-release-training-bio.txt alignment-release-training-bio
    CONVERT_DATA=alignment-release-dev-bio.txt
    ;;
sdp)
    mkdir data
    curl -L http://svn.delph-in.net/sdp/public/2015/trial/current.tgz | tar xz -C data
    python scheme/split.py -q data/sdp/trial/dm.sdp data/sdp/trial/dm
    python -m scripts.split_corpus -q data/sdp/trial/dm -t 120 -d 36 -l
    CONVERT_DATA=data/sdp/trial/*.sdp
    ;;
esac

case "$TEST_SUITE" in
unit)
    # unit tests
    python -m unittest discover -v || exit 1
    # basic parser tests
    for m in "" --sentences --paragraphs; do
      python tupa/parse.py -I 10 -t test_files/toy.xml -d test_files/toy.xml "$m" -m "model_toy$m" -v || exit 1
      python tupa/parse.py test_files/toy.xml "$m" -em "model_toy$m" -v || exit 1
    done
    python tupa/parse.py -f amr -I 10 -t test_files/LDC2014T12.txt -d test_files/LDC2014T12.txt -m model_LDC2014T12 -v || exit 1
    python tupa/parse.py -f amr test_files/LDC2014T12.txt -em model_LDC2014T12 -v || exit 1
    ;;
sparse-ucca|mlp-ucca|bilstm-ucca|noop-ucca)
    python tupa/parse.py -v -c "$ACTION" --max-words-external=5000 --layer-dim=100 -We pickle/dev/*0.pickle -t pickle/train/*0.pickle
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
convert-*)
    python scheme/convert_and_evaluate.py "$CONVERT_DATA" -v
    ;;
esac
