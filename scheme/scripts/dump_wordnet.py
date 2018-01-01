import csv
import sys

import configargparse
from nltk.corpus import wordnet as wn


def main(args):
    with open(args.rolesets_file, encoding="utf-8") as f:
        rolesets = sorted(set([l.split("-")[0] for l in f]))
    with open(args.out_file, "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        for roleset in rolesets:
            forms = related_forms(roleset)
            if forms:
                writer.writerow(map(":".join, sorted(forms)))
    print("Wrote '%s'" % args.out_file)


def related_forms(w):  # list of all derivationally related forms and their part of speech
    num_related = 0
    related = {None: w}
    while len(related) > num_related:
        num_related = len(related)
        related.update((v.synset().pos().upper(), v.name()) for x in related.values()
                       for l in wn.lemmas(x) for v in l.derivationally_related_forms())
    return [(k, v) for k, v in related.items() if k]


if __name__ == '__main__':
    argparser = configargparse.ArgParser()
    argparser.add_argument("rolesets_file", nargs="?", default="util/resources/rolesets.txt", help="rolesets to read")
    argparser.add_argument("out_file", nargs="?", default="util/resources/wordnet.txt", help="file name to write to")
    main(argparser.parse_args())
    sys.exit(0)
