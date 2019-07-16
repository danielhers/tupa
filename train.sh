#!/usr/bin/env bash
#SBATCH --mem=40G
#SBATCH --time=7-0
#SBATCH --gres=gpu:1
#SBATCH -c16

if [[ $# -lt 1 ]]; then
    SUFFIX=`date '+%Y%m%d'`
else
    SUFFIX="$1"
fi

shuf ../mrp/2019/training/training.mrp > models/mrp-${SUFFIX}.train_dev.mrp
TOTAL=`cat models/mrp-${SUFFIX}.train_dev.mrp | wc -l`
head -n$((TOTAL * 95 / 100)) models/mrp-${SUFFIX}.train_dev.mrp > models/mrp-${SUFFIX}.train.mrp
head -n$((TOTAL * 5 / 100)) models/mrp-${SUFFIX}.train_dev.mrp > models/mrp-${SUFFIX}.dev.mrp
python -m tupa --seed $RANDOM --cores=15 --use-bert --dynet-gpu \
    -t models/mrp-${SUFFIX}.train.mrp -d models/mrp-${SUFFIX}.dev.mrp \
    --conllu ../mrp/2019/companion/udpipe.mrp --alignment ../mrp/2019/companion/isi.mrp -m models/mrp-${SUFFIX}
