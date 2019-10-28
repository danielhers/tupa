#!/usr/bin/env bash
#SBATCH --mem=50G
#SBATCH --time=7-0

if [[ $# -lt 1 ]]; then
    SUFFIX=no-bert-no-dev-`date '+%Y%m%d'`
else
    SUFFIX="$1"
fi
if [[ -n "${FRAMEWORK}" ]]; then
    SUFFIX=${FRAMEWORK}-${SUFFIX}
fi

mkdir -p models

if [[ -n "${FRAMEWORK}" ]]; then
    grep '"framework": "'${FRAMEWORK}'"' ../mrp/2019/training/training.mrp > models/mrp-${SUFFIX}.train.mrp
else
    cp ../mrp/2019/training/training.mrp models/mrp-${SUFFIX}.train.mrp
fi

echo $SUFFIX
python -m tupa --seed $RANDOM --no-validate-oracle --save-every=5000 --timeout=20 \
    --dynet-autobatch --dynet-mem=25000 --dynet-check-validity \
    --max-training-per-framework=6572 -t models/mrp-${SUFFIX}.train.mrp \
    --conllu ../mrp/2019/companion/udpipe.mrp --alignment ../mrp/2019/companion/isi.mrp -m models/mrp-${SUFFIX} -v
