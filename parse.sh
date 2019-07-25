#!/usr/bin/env bash
#SBATCH --mem=50G
#SBATCH --time=1-0
#SBATCH --gres=gpu:1

if [[ $# -lt 1 ]]; then
    echo "Required: model suffix"
    exit 1
else
    SUFFIX="$1"
fi

python -m tupa --use-bert --dynet-gpu --pytorch-gpu --timeout=60 \
    ../mrp/2019/evaluation/input.mrp models/mrp-${SUFFIX}.output.mrp \
    --conllu ../mrp/2019/evaluation/udpipe.mrp --alignment ../mrp/2019/evaluation/isi.mrp -m models/mrp-${SUFFIX} -v
