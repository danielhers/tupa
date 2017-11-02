#!/usr/bin/env bash

ACTION=${TEST_SUITE%-*}
FORMAT=${TEST_SUITE#*-}
if [[ "$FORMAT" == ucca ]]; then
    SUFFIX="xml"
else
    SUFFIX="$FORMAT"
fi

# download data
if ! [[ "$ACTION" =~ ^(toy|unit)$ ]]; then
    case "$FORMAT" in
    ucca)
        mkdir pickle
        curl -L http://www.cs.huji.ac.il/~danielh/ucca/ucca_corpus_pickle.tgz | tar xz -C pickle || curl -L https://www.dropbox.com/s/q4ycn45zlmhuf9k/ucca_corpus_pickle.tgz | tar xz -C pickle
        python -m scripts.split_corpus -q pickle -t 4282 -d 454 -l
        ;;
    amr)
        curl --remote-name-all https://amr.isi.edu/download/2016-03-14/alignment-release-{training,dev,test}-bio.txt
        rename 's/.txt/.amr/' alignment-release-*-bio.txt
        python scheme/split.py -q alignment-release-training-bio.amr alignment-release-training-bio
        CONVERT_DATA=alignment-release-dev-bio.amr
        ;;
    sdp)
        mkdir data
        curl -L http://svn.delph-in.net/sdp/public/2015/trial/current.tgz | tar xz -C data
        python scheme/split.py -q data/sdp/trial/dm.sdp data/sdp/trial/dm
        python -m scripts.split_corpus -q data/sdp/trial/dm -t 120 -d 36 -l
        CONVERT_DATA=data/sdp/trial/*.sdp
        ;;
    esac
fi
export TOY_DATA="test_files/*.$SUFFIX"

case "$TEST_SUITE" in
unit-*)  # unit tests
    pytest tests --durations=0 -v tests/test_"$FORMAT".py tests/test_parser.py || exit 1
    ;;
toy-*)  # basic parser tests
    for m in "" --sentences --paragraphs; do
      python tupa/parse.py -I 10 -t "$TOY_DATA" -d "$TOY_DATA" $m -m "model_$FORMAT$m" -v || exit 1
      python tupa/parse.py "$TOY_DATA" $m -em "model_$FORMAT$m" -v || exit 1
    done
    ;;
sparse-ucca|mlp-ucca|bilstm-ucca|noop-ucca)
    python tupa/parse.py -vv -c "$ACTION" --max-words-external=5000 --layer-dim=100 -We "pickle/dev/*0.pickle" -t "pickle/train/*0.pickle"
    ;;
tune-*)
    export PARAMS_NUM=3
    while :; do
      python tupa/tune.py "$TOY_DATA" -t "$TOY_DATA" -f "$FORMAT" && break
    done
    column -t -s, params.csv
    ;;
sparse-amr)
    python tupa/parse.py -vv -c sparse --max-node-labels=250 -We "$TOY_DATA" -t "alignment-release-training-bio/*10.amr"
    ;;
noop-amr)
    python tupa/parse.py -vv -c noop -We -t alignment-release-training-bio alignment-release-dev-bio.amr
    ;;
sparse-sdp|mlp-sdp|bilstm-sdp|noop-sdp)
    python tupa/parse.py -vv -c "$ACTION" -We data/sdp/trial/dm/dev -t data/sdp/trial/dm/train
    ;;
convert-*)
    python scheme/convert_and_evaluate.py "$CONVERT_DATA" -v
    ;;
esac
