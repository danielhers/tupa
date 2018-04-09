#!/usr/bin/env python3

import csv
import os
import re
import sys
from itertools import groupby

import configargparse
from tqdm import tqdm
from ucca import evaluation, ioutil
from ucca.evaluation import LABELED, UNLABELED

from semstr.cfgutil import add_verbose_argument
from semstr.convert import CONVERTERS, UCCA_EXT
from semstr.evaluation import amr, sdp, conllu

desc = """Parses files in any format, and evaluates using the proper evaluator."""


EVALUATORS = {
    None: evaluation,
    "sdp": sdp,
    "conllu": conllu,
    "amr": amr,
}


class Scores:
    """
    Keeps score objects from multiple formats and/or languages
    """
    def __init__(self, scores):
        self.elements = [(t.aggregate(s), l) for (t, l), s in groupby(scores,
                                                                      lambda x: (type(x), getattr(x, "lang", None)))]
        element, _ = self.elements[0] if len(self.elements) == 1 else (None, None)
        self.name = element.name if element else "Multiple"
        self.format = element.format if element else None

    @staticmethod
    def aggregate(scores):
        return Scores([e for s in scores for e, _ in s.elements])

    def average_f1(self, *args, **kwargs):
        return sum(e.average_f1(*args, **kwargs) for e, _ in self.elements) / len(self.elements)

    def print(self, *args, **kwargs):
        for element, lang in self.elements:
            if len(self.elements):
                print(element.name + ((" (" + lang + ")") if lang else "") + ":", *args, **kwargs)
            element.print(*args, **kwargs)

    def fields(self):
        return [f for e, _ in self.elements for f in e.fields()]

    def titles(self):
        return [(e.name + (("_" + l) if l else "") + "_" + f) for e, l in self.elements for f in e.titles()]

    def details(self, average_f1):
        return "" if len(self.elements) < 2 else \
            " (" + ", ".join("%.3f" % average_f1(e) for e, _ in self.elements) + ")"

    def __str__(self):
        print(",".join(self.fields()))


def passage_format(filename):
    basename, ext = os.path.splitext(os.path.basename(filename))
    return basename, None if ext in UCCA_EXT else ext.lstrip(".")


def read_files(files, default_format=None):
    for filename in sorted(files, key=lambda x: tuple(map(int, re.findall("\d+", x))) or x):
        basename, converted_format = passage_format(filename)
        in_converter, out_converter = CONVERTERS.get(converted_format, CONVERTERS[default_format])
        if in_converter:
            with open(filename, encoding="utf-8") as f:
                for converted, passage, passage_id in in_converter(f, passage_id=basename, return_original=True):
                    yield converted, passage, passage_id, converted_format, in_converter, out_converter
        else:
            passage = ioutil.file2passage(filename)
            yield passage, passage, passage.ID, converted_format, in_converter, out_converter


def evaluate_all(args, evaluate, files, name=None):
    for ((guessed_converted, guessed_passage, _, guessed_format, guessed_converter, _),
         (ref_converted, ref_passage, passage_id, ref_format, _, ref_converter)) in \
            tqdm(zip(*[read_files(f, args.format) for f in files]), unit=" passages", desc=name, total=len(files[-1])):
        if not args.quiet:
            with tqdm.external_write_mode():
                print(passage_id, end=" ")
        if guessed_format != ref_format:
            guessed_passage = next(iter(guessed_converter(guessed_passage + [""], passage_id=passage_id))) if \
                ref_converter is None else ref_converter(guessed_converted)
        result = evaluate(guessed_passage, ref_passage, verbose=args.verbose > 1)
        if not args.quiet:
            with tqdm.external_write_mode():
                print("F1: %.3f" % result.average_f1(UNLABELED if args.unlabeled else LABELED))
        if args.verbose:
            with tqdm.external_write_mode():
                result.print()
        yield result


def write_csv(filename, rows):
    if filename:
        with sys.stdout if filename == "-" else open(filename, "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerows(rows)


def main(args):
    files = [[os.path.join(d, f) for f in os.listdir(d)] if os.path.isdir(d) else [d] for d in (args.guessed, args.ref)]
    evaluate = EVALUATORS.get(passage_format(files[1][0])[1], EVALUATORS[args.format]).evaluate
    results = list(evaluate_all(args, evaluate, files))
    summary = Scores(results)
    if len(results) > 1:
        if args.verbose:
            print("Aggregated scores:")
        if not args.quiet:
            print("F1: %.3f" % summary.average_f1())
            summary.print()
    elif not args.verbose:
        summary.print()
    write_csv(args.out_file,     [summary.titles()] + [result.fields() for result in results])
    write_csv(args.summary_file, [summary.titles(), summary.fields()])


if __name__ == '__main__':
    argparser = configargparse.ArgParser(description=desc)
    argparser.add_argument("guessed", help="filename/directory for the guessed annotation(s)")
    argparser.add_argument("ref", help="filename/directory for the reference annotation(s)")
    argparser.add_argument("-f", "--format", default="amr", help="default format (if cannot determine by suffix)")
    argparser.add_argument("-o", "--out-file", help="file to write results for each evaluated passage to in CSV format")
    argparser.add_argument("-s", "--summary-file", help="file to write aggregated results to, in CSV format")
    argparser.add_argument("-u", "--unlabeled", action="store_true", help="print unlabeled F1 for individual passages")
    group = argparser.add_mutually_exclusive_group()
    add_verbose_argument(group, help="detailed evaluation output")
    group.add_argument("-q", "--quiet", action="store_true", help="do not print anything")
    main(argparser.parse_args())
