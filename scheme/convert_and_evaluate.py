#!/usr/bin/env python3

import argparse
import glob
import os
import sys

from ucca import ioutil

from scheme.cfgutil import add_verbose_argument, add_boolean_option
from scheme.convert import CONVERTERS
from scheme.evaluate import EVALUATORS, Scores

desc = """Convert files to UCCA standard format, convert back to the original format and evaluate.
"""


def main():
    argparser = argparse.ArgumentParser(description=desc)
    argparser.add_argument("filenames", nargs="+", help="file names to convert and evaluate")
    add_verbose_argument(argparser, help="detailed evaluation output")
    add_boolean_option(argparser, "wikification", "use Spotlight to wikify any named node")
    argparser.add_argument("-o", "--outdir", help="output directory (if unspecified, files are not written)")
    args = argparser.parse_args()

    scores = []
    for pattern in args.filenames:
        filenames = glob.glob(pattern)
        if not filenames:
            raise IOError("Not found: " + pattern)
        for filename in filenames:
            print("\rConverting '%s'" % filename, end="")
            if args.outdir or args.verbose:
                print(flush=True)
            basename, ext = os.path.splitext(os.path.basename(filename))
            passage_format = ext.lstrip(".")
            converters = CONVERTERS.get(passage_format, CONVERTERS["amr"])
            evaluator = EVALUATORS.get(passage_format, EVALUATORS["amr"]).evaluate
            with open(filename, encoding="utf-8") as f:
                for passage, ref, passage_id in converters[0](f, passage_id=basename, return_original=True):
                    if args.outdir:
                        outfile = "%s/%s.xml" % (args.outdir, passage.ID)
                        print("Writing '%s'..." % outfile, file=sys.stderr, flush=True)
                        ioutil.passage2file(passage, outfile)
                    try:
                        guessed = converters[1](passage, wikification=args.wikification)
                    except Exception as e:
                        raise ValueError("Error converting %s back from %s" % (filename, passage_format)) from e
                    if args.outdir:
                        outfile = "%s/%s%s" % (args.outdir, passage.ID, ext)
                        print("Writing '%s'..." % outfile, file=sys.stderr, flush=True)
                        with open(outfile, "w", encoding="utf-8") as f_out:
                            print("\n".join(guessed), file=f_out)
                    try:
                        s = evaluator(guessed, ref, verbose=args.verbose > 1)
                    except Exception as e:
                        raise ValueError("Error evaluating conversion of %s" % filename) from e
                    scores.append(s)
                    if args.verbose:
                        print(passage_id)
                        s.print()
    print()
    if args.verbose and len(scores) > 1:
        print("Aggregated scores:")
    Scores(scores).print()

    sys.exit(0)


if __name__ == '__main__':
    main()
