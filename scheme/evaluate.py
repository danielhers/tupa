#!/usr/bin/env python3

import argparse
import os
from itertools import groupby

from ucca import evaluation

from scheme.cfgutil import add_verbose_argument
from scheme.convert import CONVERTERS
from scheme.evaluation import amr, sdp, conllu

desc = """Parses files in AMR format, and evaluates using smatch."""


EVALUATORS = {
    None: evaluation,
    "sdp": sdp,
    "conllu": conllu,
    "amr": amr,
}


class Scores(object):
    """
    Keeps score objects from multiple formats
    """
    def __init__(self, scores):
        self.scores_by_format = [(t, t.aggregate(s)) for t, s in groupby(scores, type)]

    @staticmethod
    def name():
        return "Multiple formats"

    @staticmethod
    def aggregate(scores):
        return Scores([s for score in scores for _, s in score.scores_by_format])

    def average_f1(self, *args, **kwargs):
        return sum(s.average_f1(*args, **kwargs) for t, s in self.scores_by_format) / len(self.scores_by_format)

    def print(self, *args, **kwargs):
        for t, s in self.scores_by_format:
            if len(self.scores_by_format):
                print(name(t) + ":", *args, **kwargs)
            s.print(*args, **kwargs)

    def fields(self):
        return [f for _, s in self.scores_by_format for f in s.fields()]

    def titles(self):
        return [(name(t) + "_" + f) for t, s in self.scores_by_format for f in s.titles()]

    def __str__(self):
        print(",".join(self.fields()))


def name(t):
    return t.name() if hasattr(t, "name") else "UCCA"


def main():
    argparser = argparse.ArgumentParser(description=desc)
    argparser.add_argument("guessed", help="file name for the guessed annotation")
    argparser.add_argument("ref", help="file name for the reference annotation")
    add_verbose_argument(argparser, help="detailed evaluation output")
    args = argparser.parse_args()

    basename, ext = os.path.splitext(os.path.basename(args.ref))
    passage_format = ext.lstrip(".")
    converter = CONVERTERS.get(passage_format, CONVERTERS["amr"])[0]
    evaluator = EVALUATORS.get(passage_format, EVALUATORS["amr"]).evaluate
    with open(args.guessed, encoding="utf-8") as guessed, open(args.ref, encoding="utf-8") as ref:
        for (guessed_passage, _), (ref_passage, passage_id) in zip(
                *[(l[1:] for l in converter(f, passage_id=basename, return_original=True)) for f in (guessed, ref)]):
            print(passage_id)
            evaluator(guessed_passage, ref_passage, verbose=args.verbose > 1).print()


if __name__ == '__main__':
    main()
