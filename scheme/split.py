#!/usr/bin/env python3
import argparse
import glob
import os
import re
import sys

from scheme.util import amr

desc = """Split sentences/passages to separate files (important for shuffling before training the parser)"""


def main(args):
    try:
        os.mkdir(args.outdir)
        print("Created " + args.outdir)
    except FileExistsError:
        pass
    lines = []
    passage_id = None
    filenames = glob.glob(args.filename)
    if not filenames:
        raise IOError("Not found: " + args.filename)
    for filename in filenames:
        _, ext = os.path.splitext(filename)
        with open(filename, encoding="utf-8") as f:
            for line in f:
                clean = line.lstrip()
                m = amr.ID_PATTERN.match(clean) or \
                    re.match("#\s*(\d+).*", line) or re.match("#\s*sent_id\s*=\s*(\S+)", line)
                if m or not clean or clean[0] != amr.COMMENT_PREFIX or re.match("#\s*::", clean):
                    lines.append(line)
                    if m:
                        passage_id = m.group(1)
                if not clean and any(map(str.strip, lines)):
                    write_file(args.outdir, passage_id, ext, lines, quiet=args.quiet)
            if lines:
                write_file(args.outdir, passage_id, ext, lines, quiet=args.quiet)
    if not args.quiet:
        print()


def write_file(outdir, passage_id, ext, lines, quiet=False):
    if passage_id is None:
        raise ValueError("Could not determine passage ID")
    filename = outdir + os.sep + passage_id + ext
    with open(filename, "w", encoding="utf-8") as f:
        f.writelines(lines)
    lines.clear()
    if not quiet:
        print("\rWrote %-70s" % filename, end="", flush=True)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=desc)
    argparser.add_argument("filename", help="file name to split")
    argparser.add_argument("outdir", help="output directory")
    argparser.add_argument("-q", "--quiet", action="store_true", help="less output")
    main(argparser.parse_args())
    sys.exit(0)
