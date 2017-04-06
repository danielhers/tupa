#!/usr/bin/env python3

import argparse
import glob
import os
import sys

from scheme import convert
from ucca.ioutil import file2passage

desc = """Parses UCCA standard format in XML or binary pickle, and writes as AMR PENMAN format.
"""


def main():
    argparser = argparse.ArgumentParser(description=desc)
    argparser.add_argument("filenames", nargs="+",
                           help="passage file names to convert")
    argparser.add_argument("-o", "--outdir", default=".",
                           help="output directory")
    argparser.add_argument("-p", "--prefix", default="",
                           help="output filename prefix")
    args = argparser.parse_args()

    for pattern in args.filenames:
        filenames = glob.glob(pattern)
        if not filenames:
            raise IOError("Not found: " + pattern)
        for filename in filenames:
            passage = file2passage(filename)
            output = convert.to_amr(passage)[0]
            outfile = "%s.txt" % (args.outdir + os.path.sep + args.prefix + passage.ID)
            sys.stderr.write("Writing '%s'...\n" % outfile)
            with open(outfile, "w", encoding="utf-8") as f:
                print(output, file=f)

    sys.exit(0)


if __name__ == '__main__':
    main()
