#!/usr/bin/env python3

import argparse
import glob
import os
import re
import sys

from ucca import convert, ioutil

from scheme.conversion.amr import from_amr, to_amr
from scheme.conversion.conllu import from_conllu, to_conllu
from scheme.conversion.sdp import from_sdp, to_sdp

desc = """Parses files in the specified format, and writes as the specified format.
Each passage is written to the file: <outdir>/<prefix><passage_id>.<extension> """


CONVERTERS = dict(convert.CONVERTERS)
CONVERTERS.update({
    "sdp":    (from_sdp,    to_sdp),
    "conllu": (from_conllu, to_conllu),
    "amr":    (from_amr,    to_amr),
})
FROM_FORMAT = {f: c[0] for f, c in CONVERTERS.items() if c[0] is not None}
TO_FORMAT = {f: c[1] for f, c in CONVERTERS.items() if c[1] is not None}

UCCA_EXT = (".xml", ".pickle")


def main(args):
    for pattern in args.filenames:
        filenames = glob.glob(pattern)
        if not filenames:
            raise IOError("Not found: " + pattern)
        for filename in filenames:
            no_ext, ext = os.path.splitext(filename)
            if ext in UCCA_EXT:  # UCCA input
                write_passage(ioutil.file2passage(filename), args)
            else:
                basename = os.path.basename(no_ext)
                try:
                    passage_id = re.search(r"\d+", basename).group(0)
                except AttributeError:
                    passage_id = basename
                converter = CONVERTERS.get(args.input_format or ext.lstrip("."))
                if converter is None:
                    raise IOError("Unknown extension '%s'. Specify format using -f" % ext)
                converter = converter[0]
                with open(filename, encoding="utf-8") as f:
                    for passage in converter(f, passage_id, split=args.split, mark_aux=args.mark_aux):
                        write_passage(passage, args)


def write_passage(passage, args):
    ext = {None: UCCA_EXT[args.binary], "amr": ".txt"}.get(args.output_format) or "." + args.output_format
    outfile = args.outdir + os.path.sep + args.prefix + passage.ID + ext
    sys.stderr.write("Writing '%s'...\n" % outfile)
    if args.output_format is None:  # UCCA output
        ioutil.passage2file(passage, outfile, args.binary)
    else:
        converter = CONVERTERS[args.output_format][1]
        output = "\n".join(converter(passage)) if args.output_format == "amr" else \
            "\n".join(line for p in (convert.split2sentences(passage) if args.split else [passage]) for line in
                      converter(p, test=args.test, tree=args.tree, mark_aux=args.mark_aux))
        with open(outfile, "w", encoding="utf-8") as f:
            print(output, file=f)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=desc)
    argparser.add_argument("filenames", nargs="+", help="file names to convert")
    argparser.add_argument("-i", "--input-format", choices=CONVERTERS, help="input file format (detected by extension)")
    argparser.add_argument("-f", "--output-format", choices=CONVERTERS, help="output file format (default: UCCA)")
    argparser.add_argument("-o", "--outdir", default=".", help="output directory")
    argparser.add_argument("-p", "--prefix", default="", help="output filename prefix")
    argparser.add_argument("-b", "--binary", action="store_true", help="write in binary format (.%s)" % UCCA_EXT[1])
    argparser.add_argument("-t", "--test", action="store_true",
                           help="omit prediction columns (head and deprel for conll; top, pred, frame, etc. for sdp)")
    argparser.add_argument("-T", "--tree", action="store_true", help="remove multiple parents to get a tree")
    argparser.add_argument("-s", "--split", action="store_true", help="split each sentence to its own passage")
    argparser.add_argument("-m", "--mark-aux", action="store_true", help="mark auxiliary edges introduced/omit edges")
    main(argparser.parse_args())
    sys.exit(0)
