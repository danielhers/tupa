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
if [[ -n "${FRAMEWORK}" ]]; then
    SUFFIX=${FRAMEWORK}-${SUFFIX}
fi

mkdir -p models
shuf ../mrp/2019/training/training.mrp > models/mrp-${SUFFIX}.train_dev.mrp

if [[ -n "${FRAMEWORK}" ]]; then
    grep '"framework": "'${FRAMEWORK}'"' models/mrp-${SUFFIX}.train_dev.mrp > models/mrp-${SUFFIX}.train_dev.mrp.temp
    mv models/mrp-${SUFFIX}.train_dev.mrp.temp models/mrp-${SUFFIX}.train_dev.mrp
fi

TOTAL=`cat models/mrp-${SUFFIX}.train_dev.mrp | wc -l`
head -n$((TOTAL * 95 / 100)) models/mrp-${SUFFIX}.train_dev.mrp > models/mrp-${SUFFIX}.train.mrp
head -n$((TOTAL * 5 / 100)) models/mrp-${SUFFIX}.train_dev.mrp > models/mrp-${SUFFIX}.dev.mrp

python -m tupa --seed $RANDOM --cores=15 --use-bert --pytorch-gpu --no-validate-oracle \
    -t models/mrp-${SUFFIX}.train.mrp -d models/mrp-${SUFFIX}.dev.mrp \
    --conllu ../mrp/2019/companion/udpipe.mrp --alignment ../mrp/2019/companion/isi.mrp -m models/mrp-${SUFFIX} -v
