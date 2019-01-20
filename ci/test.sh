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
        curl -L https://github.com/UniversalConceptualCognitiveAnnotation/UCCA_English-Wiki/releases/download/v1.2.4/ucca-sample.tar.gz | tar xz -C pickle
        TRAIN_DATA="pickle/train/*"
        DEV_DATA="pickle/dev/*"
        ;;
    amr)
        curl -L --remote-name-all https://amr.isi.edu/download/2016-03-14/alignment-release-{training,dev,test}-bio.txt
        rename 's/.txt/.amr/' alignment-release-*-bio.txt
        CONVERT_DATA=alignment-release-dev-bio.amr
        TRAIN_DATA=alignment-release-training-bio.amr
        DEV_DATA=alignment-release-dev-bio.amr
        ;;
    sdp)
        mkdir data
        curl -L http://svn.delph-in.net/sdp/public/2015/trial/current.tgz | tar xz -C data
        python -m semstr.scripts.split -q data/sdp/trial/dm.sdp -o data/sdp/trial/dm
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
      echo Training with ${args}
      tupa -c sparse -I 10 -t "$TOY_DATA" -d "$TOY_DATA" ${args} || exit 1
      echo Testing with ${args}
      tupa "$TOY_DATA" -e ${args} || exit 1
      echo Testing on text file with ${args}
      tupa test_files/example.txt ${args} || exit 1
    done
    ALL_DATA="test_files/*.xml test_files/*.sdp test_files/*.amr test_files/*.conllu"
    tupa -We ${ALL_DATA} -t ${ALL_DATA} -m multilingual --layer-dim=2 --lstm-layer-dim=2 --pos-dim=2 \
        --lemma-dim=2 --tag-dim=2 --word-dim-external=0 --dep-dim=2 --edge-label-dim=2 --ner-dim=2 --node-label-dim=2 \
        --iterations 5=--optimizer=sgd 10=--optimizer=adam --eval-test --hyperparam de=--lemma-dim=4 --multilingual -v \
        -u "$FORMAT" --timeout=1 || exit 1
    tupa -We ${ALL_DATA} -m multilingual --timeout=1 || exit 1
    python -m tupa.scripts.strip_multitask multilingual || exit 1
    tupa -We ${TOY_DATA} -m multilingual --timeout=1 || exit 1
    ;;
tune-*)
    export PARAMS_NUM=3 MAX_ITERATIONS=3
    while :; do
      python -m tupa.scripts.tune "$TOY_DATA" -t "$TOY_DATA" -f "$FORMAT" --max-action-ratio 10 && break
      echo Retrying...
      rm -fv models/*
    done
    column -t -s, params.csv
    ;;
noop-amr)
    tupa -vv -c noop --implicit -We -I 1 -t "$TRAIN_DATA" "$DEV_DATA"
    ;;
*-amr)
    tupa -vv -c "$ACTION" --implicit -We "$TOY_DATA" -I 1 -t "$TRAIN_DATA" --max-node-labels=250 --max-training-per-format=100
    ;;
*)
    echo Training on "$TRAIN_DATA"
    tupa -vv -c "$ACTION" -We "$DEV_DATA" -I 1 -t "$TRAIN_DATA" --max-words-external=5000 --word-dim=100 --lstm-layer-dim=100 --embedding-layer-dim=100 || exit 1
    echo Testing on "$DEV_DATA"
    tupa -vv -m "$ACTION" -We "$DEV_DATA"
    ;;
esac
