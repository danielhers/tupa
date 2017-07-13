#!/usr/bin/env python3

import argparse

from scheme.evaluation.amr import evaluate
from tupa.config import VAction

desc = """Parses files in AMR format, and evaluates using smatch.
"""


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
    argparser.add_argument("-v", "--verbose", nargs="?", action=VAction, default=0, help="detailed evaluation output")
    args = argparser.parse_args()

    with open(args.guessed, encoding="utf-8") as guessed, open(args.ref, encoding="utf-8") as ref:
        for g, r in zip(read_amr(guessed), read_amr(ref)):
            evaluate(g, r, verbose=args.verbose > 1).print()


if __name__ == '__main__':
    main()
