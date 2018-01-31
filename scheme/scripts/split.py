#!/usr/bin/env python3
import glob
import os
import re

import configargparse

from scheme.util.amr import ID_PATTERN, COMMENT_PREFIX

desc = """Split sentences/passages to separate files (important for shuffling before training the parser)"""


def main(args):
    try:
        os.mkdir(args.outdir)
        print("Created " + args.outdir)
    except FileExistsError:
        pass
    lines = []
    passage_id = 0
    doc_id = None
    filenames = glob.glob(args.filename)
    if not filenames:
        raise IOError("Not found: " + args.filename)
    for filename in filenames:
        _, ext = os.path.splitext(filename)
        with open(filename, encoding="utf-8") as f:
            for line in f:
                clean = line.lstrip()
                m_id = ID_PATTERN.match(clean) or \
                    re.match("#\s*(\d+).*", line) or re.match("#\s*sent_id\s*=\s*(\S+)", line)
                m_docid = re.match("#\s*doc_id\s*=\s*(\S+)", line)
                if m_id or m_docid or not clean or clean[0] != COMMENT_PREFIX or re.match("#\s*::", clean):
                    lines.append(line)
                    if m_docid:
                        doc_id = m_docid.group(1)
                        passage_id = 1
                    if m_id:
                        passage_id = m_id.group(1)
                if not clean and any(map(str.strip, lines)):
                    if not args.doc_ids or doc_id in args.doc_ids:
                        write_file(args.outdir, doc_id, passage_id, ext, lines, quiet=args.quiet)
                    lines.clear()
                    if isinstance(passage_id, str):
                        passage_id = None
                    else:
                        passage_id += 1
            if lines and (not args.doc_ids or doc_id in args.doc_ids):
                write_file(args.outdir, doc_id, passage_id, ext, lines, quiet=args.quiet)
    if not args.quiet:
        print()


def write_file(outdir, doc_id, passage_id, ext, lines, quiet=False):
    if passage_id is None:
        raise ValueError("Could not determine passage ID")
    filename = os.path.join(outdir, ("" if doc_id is None else (doc_id + ".")) + str(passage_id) + ext)
    with open(filename, "w", encoding="utf-8") as f:
        f.writelines(lines)
    if not quiet:
        print("\rWrote %-70s" % filename, end="", flush=True)


if __name__ == '__main__':
    argparser = configargparse.ArgParser(description=desc)
    argparser.add_argument("filename", help="file name to split")
    argparser.add_argument("outdir", help="output directory")
    argparser.add_argument("-q", "--quiet", action="store_true", help="less output")
    argparser.add_argument("--doc-ids", nargs="+", help="document IDs to keep from the input file (by '# doc_id')")
    main(argparser.parse_args())
