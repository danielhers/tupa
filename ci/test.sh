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
        curl -L http://www.cs.huji.ac.il/~danielh/ucca/ucca-sample.tar.gz | tar xz -C pickle
        TRAIN_DATA="pickle/train/*"
        DEV_DATA="pickle/dev/*"
        ;;
    amr)
        curl --remote-name-all https://amr.isi.edu/download/2016-03-14/alignment-release-{training,dev,test}-bio.txt
        rename 's/.txt/.amr/' alignment-release-*-bio.txt
        python -m semstr.scripts.split -q alignment-release-training-bio.amr alignment-release-training-bio
        CONVERT_DATA=alignment-release-dev-bio.amr
        TRAIN_DATA=alignment-release-training-bio
        DEV_DATA=alignment-release-dev-bio.amr
        ;;
    sdp)
        mkdir data
        curl -L http://svn.delph-in.net/sdp/public/2015/trial/current.tgz | tar xz -C data
        python -m semstr.scripts.split -q data/sdp/trial/dm.sdp data/sdp/trial/dm
        python -m scripts.split_corpus -q data/sdp/trial/dm -t 120 -d 36 -l
        CONVERT_DATA=data/sdp/trial/*.sdp
        TRAIN_DATA=data/sdp/trial/dm/train
        DEV_DATA=data/sdp/trial/dm/dev
        ;;
    esac
fi
export TOY_DATA="test_files/*.$SUFFIX"

case "$TEST_SUITE" in
unit)  # unit tests
    pytest --durations=0 -v tests ${TEST_OPTIONS} || exit 1
    ;;
toy-*)  # basic parser tests
    for m in "" --sentences --paragraphs; do
      args="$m -m model_$FORMAT$m -v"
      tupa -c sparse -I 10 -t "$TOY_DATA" -d "$TOY_DATA" ${args} || exit 1
      tupa "$TOY_DATA" -e ${args} || exit 1
      tupa test_files/example.txt ${args} || exit 1
    done
    ;;
tune-*)
    export PARAMS_NUM=3 MAX_ITERATIONS=3
    while :; do
      python -m tupa.scripts.tune "$TOY_DATA" -t "$TOY_DATA" -f "$FORMAT" --max-action-ratio 10 && break
      rm -fv models/*
    done
    column -t -s, params.csv
    ;;
noop-amr)
    tupa -vv -c noop --implicit -We -I 1 -t "$TRAIN_DATA" "$DEV_DATA"
    ;;
*-amr)
    tupa -vv -c "$ACTION" --implicit -We "$TOY_DATA" -I 1 -t "alignment-release-training-bio/*10.amr" --max-node-labels=250
    ;;
*)
    tupa -vv -c "$ACTION" -We "$DEV_DATA" -I 1 -t "$TRAIN_DATA" --max-words-external=5000 --word-dim=100 --lstm-layer-dim=100 --embedding-layer-dim=100 || exit 1
    tupa -vv -m "$ACTION" -We "$DEV_DATA"
    ;;
esac
