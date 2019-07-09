#!/usr/bin/env python3

import argparse
import csv
from operator import attrgetter

from spacy.symbols import NAMES
from tqdm import tqdm

from tupa.features.feature_params import load_spacy_model

desc = """Create vocabulary file from spaCy model."""


def main(args):
    nlp = load_spacy_model(args.model_name)
    vocab = list(enumerate(NAMES)) + \
        [(l.orth, l.orth_) for l in sorted(nlp.vocab, key=attrgetter("prob"), reverse=True)]
    out_file = args.out_file or args.model_name + ".csv"
    with open(out_file, "w", encoding="utf-8") as f:
        csv.writer(f).writerows(tqdm(vocab, unit=" words", desc="Writing " + out_file))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=desc)
    argparser.add_argument("model_name", nargs="?", help="name of spaCy model", default="en")
    argparser.add_argument("out_file", nargs="?", help="filename to create")
    argparser.add_argument("-l", "--lang", default="en", help="small two-letter language code to use for NLP model")
    main(argparser.parse_args())
