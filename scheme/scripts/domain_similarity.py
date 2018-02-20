import csv
from collections import Counter
from itertools import chain

import configargparse
import numpy as np
from scipy.stats import entropy
from tqdm import tqdm
from ucca import layer0
from ucca.ioutil import read_files_and_dirs

desc = """Calculate similarity between dataset domains."""


def main(args):
    p = {d: probs(d) for d in args.dirs}
    vocab = sorted(set(chain(*p.values())))
    reps = np.array([[p[d].get(v, 0.0) for v in vocab] for d in args.dirs], dtype=float)
    sims = np.array([i * [np.nan] + [sim(reps[i], reps[j], args.similarity) for j in range(i, len(reps))]
                     for i in range(len(reps))], dtype=float)
    print(sims)
    filename = "similarities.txt"
    np.savetxt(filename, sims)
    print("Saved '%s'" % filename)


def sim(x, y, similarity):
    if similarity == "var":
        return np.linalg.norm(x - y, ord=1)
    elif similarity == "cos":
        return x.dot(y) / (np.linalg.norm(x) * np.linalg.norm(y))
    elif similarity == "euc":
        return np.linalg.norm(x - y)
    elif similarity == "js":
        return (entropy(x, (x + y) / 2) + entropy(y, (x + y) / 2)) / 2
    raise ValueError("Unknown similarity '%s'" % similarity)


def probs(d):
    filename = d + ".freq.csv"
    try:
        with open(filename, encoding="utf-8") as f:
            counts = dict((key, int(value)) for key, value in csv.reader(f))
        print("Loaded '%s'" % filename)
    except IOError:
        counts = Counter()
        for p in tqdm(read_files_and_dirs(d), unit=" passages", desc="Reading %s" % d):
            for t in p.layer(layer0.LAYER_ID).all:
                counts[t.text] += 1
        with open(filename, "w", encoding="utf-8") as f:
            csv.writer(f).writerows(counts.most_common())
        print("Saved '%s'" % filename)
    s = sum(counts.values())
    return {key: float(value) / s for key, value in counts.items()}


if __name__ == '__main__':
    argparser = configargparse.ArgParser(description=desc)
    argparser.add_argument("dirs", nargs="+", help="directories with passages to compare")
    argparser.add_argument("-s", "--similarity", choices=("var", "cos", "euc", "js"), default="var")
    main(argparser.parse_args())
