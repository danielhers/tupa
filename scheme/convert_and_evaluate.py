#!/usr/bin/env python3

import argparse
import glob
import sys

import os
from ucca.ioutil import passage2file

from scheme.conversion.amr import from_amr, to_amr
from scheme.evaluation.amr import evaluate, Scores
from tupa.config import VAction

desc = """Parses files in AMR format, converts to UCCA standard format,
converts back to the original format and evaluates using smatch.
"""


def main():
    argparser = argparse.ArgumentParser(description=desc)
    argparser.add_argument("filenames", nargs="+", help="file names to convert and evaluate")
    argparser.add_argument("-v", "--verbose", nargs="?", action=VAction, default=0, help="detailed evaluation output")
    argparser.add_argument("-o", "--outdir", help="output directory (if unspecified, files are not written)")
    args = argparser.parse_args()

    scores = []
    for pattern in args.filenames:
        filenames = glob.glob(pattern)
        if not filenames:
            raise IOError("Not found: " + pattern)
        for filename in filenames:
            sys.stdout.write("\rConverting '%s'" % filename)
            if args.outdir:
                sys.stdout.write("\n")
            sys.stdout.flush()
            basename = os.path.basename(os.path.splitext(filename)[0])
            with open(filename, encoding="utf-8") as f:
                for passage, ref, amr_id in from_amr(f, passage_id=basename, return_amr=True):
                    if args.outdir:
                        outfile = "%s/%s.xml" % (args.outdir, passage.ID)
                        print("Writing '%s'..." % outfile, file=sys.stderr, flush=True)
                        passage2file(passage, outfile)
                    try:
                        guessed = to_amr(passage, amr_id)[0]
                    except Exception as e:
                        raise ValueError("Error converting %s back from AMR" % filename) from e
                    if args.outdir:
                        outfile = "%s/%s.txt" % (args.outdir, passage.ID)
                        print("Writing '%s'..." % outfile, file=sys.stderr, flush=True)
                        with open(outfile, "w", encoding="utf-8") as f_out:
                            print(str(guessed), file=f_out)
                    try:
                        s = evaluate(guessed, ref, verbose=args.verbose > 1)
                    except Exception as e:
                        raise ValueError("Error evaluating conversion of %s" % filename) from e
                    scores.append(s)
                    if args.verbose:
                        s.print(flush=True)
    print()
    if args.verbose and len(scores) > 1:
        print("Aggregated scores:")
    Scores.aggregate(scores).print()

    sys.exit(0)


if __name__ == '__main__':
    main()
