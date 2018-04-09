import csv
import os
import sys
from shutil import rmtree, move

import configargparse
import nltk

EXTRA_ROLESETS = (  # fix for roles missing in PropBank
    ("ablate-01", "0", "1", "2", "3"),
    ("play-11", "0", "1", "2", "3"),
    ("raise-02", "0", "1", "2", "3"),
)


def main(args):
    rolesets = install_propbank() or get_rolesets()
    with open(args.out_file, "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        for roleset in rolesets:
            try:
                writer.writerow([roleset.attrib["id"].replace(".", "-")] +
                                sorted([r.attrib["n"] for r in roleset.findall("roles/role")], key=int))
            except ValueError:
                print("Non numeric role for " + roleset.attrib["id"])
        writer.writerows(EXTRA_ROLESETS)
    print("Wrote '%s'" % args.out_file)


def install_propbank():
    try:
        return get_rolesets()
    except (LookupError, OSError):
        print("Getting PropBank...")
        corpora_dir = os.path.join(nltk.data.path[0], "corpora")
        target_frames_dir = os.path.join(corpora_dir, "propbank", "frames")
        propbank_git_dir = os.path.join(corpora_dir, "propbank-frames")
        nltk.downloader.Downloader().download("propbank")
        rmtree(propbank_git_dir, ignore_errors=True)
        os.system("git clone https://github.com/propbank/propbank-frames " + propbank_git_dir)
        rmtree(target_frames_dir, ignore_errors=True)
        move(os.path.join(propbank_git_dir, "frames"), target_frames_dir)


def get_rolesets():
    return nltk.corpus.propbank.rolesets()


if __name__ == '__main__':
    argparser = configargparse.ArgParser()
    argparser.add_argument("out_file", nargs="?", default="util/resources/rolesets.txt", help="file name to write to")
    main(argparser.parse_args())
    sys.exit(0)
