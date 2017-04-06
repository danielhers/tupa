#!/usr/bin/env python3

import argparse
import glob
import os
import re
import sys

from conversion.amr import CONVERTERS
from ucca.ioutil import passage2file

desc = """Parses files in the specified format, and writes UCCA standard format, as XML or binary pickle.
Each passage is written to the file:
<outdir>/<prefix><passage_id>.<extension>
"""


def main():
    argparser = argparse.ArgumentParser(description=desc)
    argparser.add_argument("filenames", nargs="+", help="file names to convert")
    argparser.add_argument("-f", "--format", choices=CONVERTERS, help="input file format")
    argparser.add_argument("-o", "--outdir", default=".", help="output directory")
    argparser.add_argument("-p", "--prefix", default="", help="output filename prefix")
    argparser.add_argument("-b", "--binary", action="store_true", help="write in pickle binary format (.pickle)")
    argparser.add_argument("-s", "--split", action="store_true", help="split each sentence to its own passage")
    argparser.add_argument("-T", "--tree", action="store_true", help="currently unused")
    argparser.add_argument("-m", "--mark-aux", action="store_true", help="mark auxiliary edges introduced")
    args = argparser.parse_args()

    for pattern in args.filenames:
        filenames = glob.glob(pattern)
        if not filenames:
            raise IOError("Not found: " + pattern)
        for filename in filenames:
            no_ext, ext = os.path.splitext(filename)
            basename = os.path.basename(no_ext)
            try:
                passage_id = re.search(r"\d+", basename).group(0)
            except AttributeError:
                passage_id = basename
            converter = CONVERTERS.get(args.format or ext.lstrip("."))[0]
            if converter is None:
                raise IOError("Unknown extension '%s'. Specify format using -f" % ext)
            with open(filename, encoding="utf-8") as f:
                for passage in converter(f, passage_id, split=args.split, mark_aux=args.mark_aux):
                    outfile = "%s/%s.%s" % (args.outdir, args.prefix + passage.ID,
                                            "pickle" if args.binary else "xml")
                    sys.stderr.write("Writing '%s'...\n" % outfile)
                    passage2file(passage, outfile, args.binary)

    sys.exit(0)


if __name__ == '__main__':
    main()
