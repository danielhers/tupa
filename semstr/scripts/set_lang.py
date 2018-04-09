#!/usr/bin/env python3

import argparse

from tqdm import tqdm
from ucca.ioutil import write_passage

from semstr.scripts.annotate import add_specs_args, read_specs

desc = """Read passages in any format, and write back with attrib['lang'] set."""


def main(args):
    for passages, out_dir, lang in read_specs(args):
        for passage in tqdm(passages, unit=" passages", desc="Setting language in " + out_dir, postfix={"lang": lang}):
            passage.attrib["lang"] = lang
            write_passage(passage, outdir=out_dir, verbose=False, binary=args.binary)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=desc)
    add_specs_args(argparser)
    main(argparser.parse_args())
