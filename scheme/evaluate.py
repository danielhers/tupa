#!/usr/bin/env python3

import os
from itertools import groupby

import configargparse
from ucca import evaluation, ioutil

from scheme.cfgutil import add_verbose_argument
from scheme.convert import CONVERTERS, UCCA_EXT
from scheme.evaluation import amr, sdp, conllu

desc = """Parses files in any format, and evaluates using the proper evaluator."""


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


def read_passages(filename, default_format=None):
    basename, converted_format = passage_format(filename)
    in_converter, out_converter = CONVERTERS.get(converted_format, CONVERTERS[default_format])
    if in_converter:
        with open(filename, encoding="utf-8") as f:
            for _, passage, passage_id in in_converter(f, passage_id=basename, return_original=True):
                yield passage, passage_id, converted_format, in_converter, out_converter
    else:
        passage = ioutil.file2passage(filename)
        yield passage, passage.ID, converted_format, in_converter, out_converter


def passage_format(filename):
    basename, ext = os.path.splitext(os.path.basename(filename))
    return basename, None if ext in UCCA_EXT else ext.lstrip(".")


def main(args):
    evaluate = EVALUATORS.get(passage_format(args.ref)[1], EVALUATORS[args.format]).evaluate
    for (guessed_passage, _, guessed_format, guessed_converter, _), \
        (ref_passage, passage_id, ref_format, _, ref_converter) in zip(*[read_passages(filename, args.format)
                                                                         for filename in (args.guessed, args.ref)]):
        print(passage_id)
        if guessed_format != ref_format:
            guessed_passage = next(iter(guessed_converter(guessed_passage + [""], passage_id=passage_id))) if \
                ref_converter is None else ref_converter(guessed_passage)
        evaluate(guessed_passage, ref_passage, verbose=args.verbose > 1).print()


if __name__ == '__main__':
    argparser = configargparse.ArgParser(description=desc)
    argparser.add_argument("guessed", help="file name for the guessed annotation")
    argparser.add_argument("ref", help="file name for the reference annotation")
    argparser.add_argument("-f", "--format", default="amr", help="default format (if cannot determine by suffix)")
    add_verbose_argument(argparser, help="detailed evaluation output")
    main(argparser.parse_args())
