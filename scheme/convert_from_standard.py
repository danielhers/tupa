#!/usr/bin/env python3

import argparse
import glob
import os
import sys

from conversion.amr import CONVERTERS
from ucca.convert import split2sentences
from ucca.ioutil import file2passage

desc = """Parses UCCA standard format in XML or binary pickle, and writes as the specified format."""


def main():
    argparser = argparse.ArgumentParser(description=desc)
    argparser.add_argument("filenames", nargs="+", help="passage file names to convert")
    argparser.add_argument("-f", "--format", choices=CONVERTERS, required=True, help="output file format")
    argparser.add_argument("-o", "--outdir", default=".", help="output directory")
    argparser.add_argument("-p", "--prefix", default="", help="output filename prefix")
    argparser.add_argument("-t", "--test", action="store_true",
                           help="omit prediction columns (head and deprel for conll; top, pred, frame, etc. for sdp)")
    argparser.add_argument("-s", "--sentences", action="store_true", help="split passages to sentences")
    argparser.add_argument("-T", "--tree", action="store_true", help="remove multiple parents to get a tree")
    argparser.add_argument("-m", "--mark-aux", action="store_true", help="omit marked auxiliary edges")
    args = argparser.parse_args()

    converter = CONVERTERS[args.format][1]
    for pattern in args.filenames:
        filenames = glob.glob(pattern)
        if not filenames:
            raise IOError("Not found: " + pattern)
        for filename in filenames:
            passage = file2passage(filename)
            if args.format == "amr":
                output, ext = converter(passage)[0], "txt"
            else:
                passages = split2sentences(passage) if args.sentences else [passage]
                output = "\n".join(line for p in passages for line in
                                   converter(p, test=args.test, tree=args.tree, mark_aux=args.mark_aux))
                ext = args.format
            outfile = args.outdir + os.path.sep + args.prefix + passage.ID + "." + ext
            sys.stderr.write("Writing '%s'...\n" % outfile)
            with open(outfile, "w", encoding="utf-8") as f:
                print(output, file=f)

    sys.exit(0)


if __name__ == '__main__':
    main()
