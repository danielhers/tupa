#!/usr/bin/env bash
#SBATCH --mem=50G
#SBATCH --time=7-0
#SBATCH -c2

if [[ $# -lt 1 ]]; then
    SUFFIX=no-bert-`date '+%Y%m%d'`
else
    SUFFIX="$1"
fi
if [[ -n "${FRAMEWORK}" ]]; then
    SUFFIX=${FRAMEWORK}-${SUFFIX}
fi

mkdir -p models
for FRAMEWORK in ${FRAMEWORK:-`grep -Po '(?<="framework": ")\w+(?=")' ../mrp/2019/training/training.mrp | sort -u`}; do
    grep '"framework": "'${FRAMEWORK}'"' ../mrp/2019/training/training.mrp | shuf > models/mrp-${SUFFIX}.train_dev.${FRAMEWORK}.mrp
done
head -n 500 -q models/mrp-${SUFFIX}.train_dev.*.mrp > models/mrp-${SUFFIX}.dev.mrp
tail -n+501 -q models/mrp-${SUFFIX}.train_dev.*.mrp | shuf > models/mrp-${SUFFIX}.train.mrp
rm -f models/mrp-${SUFFIX}.train_dev.*.mrp

echo $SUFFIX
python -m tupa --seed $RANDOM --cores=2 --no-validate-oracle --save-every=50000 --timeout=20 \
    --dynet-autobatch --dynet-mem=15000 --dynet-check-validity \
    --max-training-per-framework=6572 -t models/mrp-${SUFFIX}.train.mrp -d models/mrp-${SUFFIX}.dev.mrp \
    --conllu ../mrp/2019/companion/udpipe.mrp --alignment ../mrp/2019/companion/isi.mrp -m models/mrp-${SUFFIX} -v
