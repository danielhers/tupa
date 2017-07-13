#!/usr/bin/env python3
import argparse
import glob
import sys

from scheme.util.amr import *

desc = """Split AMRs to separate files (important for shuffling before training the parser)"""


def main(args):
    try:
        os.mkdir(args.outdir)
        print("Created " + args.outdir)
    except FileExistsError:
        pass
    lines = []
    amr_id = None
    filenames = glob.glob(args.filename)
    if not filenames:
        raise IOError("Not found: " + args.filename)
    for filename in filenames:
        with open(filename, encoding="utf-8") as f:
            for line in f:
                clean = line.lstrip()
                if clean:
                    if clean[0] != COMMENT_PREFIX or re.match("#\s*::", clean):
                        lines.append(line)
                        m = re.match(ID_PATTERN, clean)
                        if m:
                            amr_id = m.group(1)
                elif lines:
                    write_file(args.outdir, amr_id, lines, quiet=args.quiet)
            if lines:
                write_file(args.outdir, amr_id, lines, quiet=args.quiet)


def write_file(outdir, amr_id, lines, quiet=False):
    filename = outdir + os.sep + amr_id + ".txt"
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
