#!/usr/bin/env bash
#SBATCH --mem=20G
#SBATCH --time=1-0

python -m tupa -I1 -c noop \
    -t ../mrp/2019/training/training.mrp \
    --conllu ../mrp/2019/companion/udpipe.mrp --alignment ../mrp/2019/companion/isi.mrp -m models/mrp-noop -v
