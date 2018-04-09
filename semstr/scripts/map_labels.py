#!/usr/bin/env python3

import csv
from argparse import ArgumentParser

from tqdm import tqdm
from ucca import ioutil
from ucca.evaluation import evaluate, Scores, LABELED, PRIMARY

from semstr.convert import CONVERTERS

desc = "Create confusion matrix of labels between two datasets, " \
       "and use it to create a CSV file mapping labels by most-frequent"


def main(args):
    guessed, ref = [ioutil.read_files_and_dirs((x,), converters=CONVERTERS) for x in (args.guessed, args.ref)]
    if len(guessed) != len(ref):
        raise ValueError("Number of passages to compare does not match: %d != %d" % (len(guessed), len(ref)))
    if len(guessed) > 1:
        guessed_by_id = {g.ID: g for g in tqdm(guessed, desc="Reading " + args.guessed, unit=" passages")}
        try:
            guessed = [guessed_by_id[p.ID] for p in tqdm(ref, desc="Reading " + args.ref, unit=" passages")]
        except KeyError as e:
            raise ValueError("Passage IDs do not match") from e
    results = [evaluate(g, r, errors=True) for g, r in zip(tqdm(guessed, desc="Evaluating", unit=" passages"), ref)]
    confusion_matrix = Scores.aggregate(results).evaluators[LABELED].results[PRIMARY].errors.most_common()
    label_map = {}
    for (g, r), _ in confusion_matrix:
        g, _ = g.partition("|")
        if g not in label_map:
            label_map[g] = r.partition("|")[0]
    with open(args.out_file, "w", encoding="utf-8") as f:
        csv.writer(f).writerows(tqdm(sorted(label_map.items()), desc="Writing " + args.out_file, unit=" rows"))


if __name__ == "__main__":
    argparser = ArgumentParser(description=desc)
    argparser.add_argument("guessed", help="filename for the guessed annotation, or directory of files")
    argparser.add_argument("ref", help="xml/pickle filename for the reference annotation, or directory of files")
    argparser.add_argument("-o", "--out-file", default="label_map.csv", help="output CSV file")
    main(argparser.parse_args())
