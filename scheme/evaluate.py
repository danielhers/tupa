#!/usr/bin/env python3

import argparse
from itertools import groupby

from ucca import evaluation

from scheme.cfgutil import add_verbose_argument
from scheme.evaluation import amr, sdp

desc = """Parses files in AMR format, and evaluates using smatch."""


EVALUATORS = {
    None: evaluation,
    "sdp": sdp,
    "amr": amr,
}


class Scores(object):
    """
    Keeps score objects from multiple formats
    """
    def __init__(self, scores):
        self.scores_by_format = [(t, t.aggregate(s)) for t, s in groupby(scores, type)]

    def average_f1(self, *args, **kwargs):
        return sum(s.average_f1(*args, **kwargs) for t, s in self.scores_by_format) / len(self.scores_by_format)

    def print(self, *args, **kwargs):
        for _, s in self.scores_by_format:
            s.print(*args, **kwargs)

    def fields(self):
        return [f for _, s in self.scores_by_format for f in s.fields()]

    def titles(self):
        return [t for _, s in self.scores_by_format for t in s.titles()]

    def __str__(self):
        print(",".join(self.fields()))


def read_amr(f):
    lines = []
    for line in f:
        line = line.lstrip()
        if line:
            if line[0] != "#":
                lines.append(line)
                continue
        if lines:
            yield " ".join(lines)
    if lines:
        yield " ".join(lines)


def main():
    argparser = argparse.ArgumentParser(description=desc)
    argparser.add_argument("guessed", help="file name for the guessed annotation")
    argparser.add_argument("ref", help="file name for the reference annotation")
    add_verbose_argument(argparser, help="detailed evaluation output")
    args = argparser.parse_args()

    with open(args.guessed, encoding="utf-8") as guessed, open(args.ref, encoding="utf-8") as ref:
        for g, r in zip(read_amr(guessed), read_amr(ref)):
            EVALUATORS["amr"].evaluate(g, r, verbose=args.verbose > 1).print()


if __name__ == '__main__':
    main()
