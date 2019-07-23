#!/usr/bin/env bash
#SBATCH --mem=20G
#SBATCH --time=1-0

if [[ $# -lt 1 ]]; then
    echo "Required: model suffix"
    exit 1
else
    SUFFIX="$1"
fi

python -m tupa ../mrp/2019/evaluation/input.mrp models/mrp-${SUFFIX}.output.mrp \
    --conllu ../mrp/2019/evaluation/udpipe.mrp --alignment ../mrp/2019/evaluation/isi.mrp -m models/mrp-${SUFFIX} -v
