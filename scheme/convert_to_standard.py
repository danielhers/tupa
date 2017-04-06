#!/usr/bin/env python3

import argparse
import glob
import os
import re
import sys

from conversion.amr import from_amr
from ucca.ioutil import passage2file

desc = """Parses files in AMR PENMAN format, and writes UCCA standard format, as XML or binary pickle.
Each passage is written to the file:
<outdir>/<prefix><passage_id>.<extension>
"""


def main():
    argparser = argparse.ArgumentParser(description=desc)
    argparser.add_argument("filenames", nargs="+",
                           help="file names to convert")
    argparser.add_argument("-o", "--outdir", default=".",
                           help="output directory")
    argparser.add_argument("-p", "--prefix", default="",
                           help="output filename prefix")
    argparser.add_argument("-b", "--binary", action="store_true",
                           help="write in pickle binary format (.pickle)")
    args = argparser.parse_args()

    for pattern in args.filenames:
        filenames = glob.glob(pattern)
        if not filenames:
            raise IOError("Not found: " + pattern)
        for filename in filenames:
            no_ext, _ = os.path.splitext(filename)
            basename = os.path.basename(no_ext)
            try:
                passage_id = re.search(r"\d+", basename).group(0)
            except AttributeError:
                passage_id = basename

            with open(filename, encoding="utf-8") as f:
                for passage in from_amr(f, passage_id):
                    outfile = "%s/%s.%s" % (args.outdir, args.prefix + passage.ID,
                                            "pickle" if args.binary else "xml")
                    sys.stderr.write("Writing '%s'...\n" % outfile)
                    passage2file(passage, outfile, args.binary)

    sys.exit(0)


if __name__ == '__main__':
    main()
