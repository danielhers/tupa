#!/usr/bin/env bash
#SBATCH --mem=50G
#SBATCH --time=7-0
#SBATCH --gres=gpu:1
#SBATCH -c16

if [[ $# -lt 1 ]]; then
    SUFFIX=`date '+%Y%m%d'`
else
    SUFFIX="$1"
fi

python -m tupa --use-bert --dynet-gpu \
    ../mrp/2019/evaluation/input.mrp \
    --conllu ../mrp/2019/evaluation/udpipe.mrp --alignment ../mrp/2019/evaluation/isi.mrp -m models/mrp-${SUFFIX}
