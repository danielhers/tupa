import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import re
import os
from argparse import ArgumentParser
from scipy.misc import imresize
from matplotlib.ticker import MaxNLocator

np.seterr("raise")


def smooth(x, s=100):
    x = x.reshape((x.shape[0], -1))
    return imresize(x, size=(min(x.shape[0], s), min(x.shape[1], s)))


def main(args):
    for filename in args.data:
        basename, _ = os.path.splitext(filename)
        print(filename)
        values = {}
        name = None
        with open(filename) as f:
            for line in f:
                if line.startswith("#"):
                    name = line
                    print(name.strip())
                else:
                    _, key, shape, *_ = name.split()
                    key = tuple(re.findall(r"(?<=/)[a-z]+(?=/)|(?<=_)\d+", key))
                    if len(key) == 1:
                        key = ("",) + key
                    key = (key[0],) + tuple(map(int, key[1:]))
                    shape = tuple(map(int, re.findall(r"\d+", shape)))
                    values[key] = smooth(np.reshape(np.fromstring(line, sep=" "), shape), s=args.smoothing)
        _, axes = plt.subplots(len(values), figsize=(19, 2 * len(values)))
        plt.tight_layout()
        for i, (n, v) in enumerate(sorted(values.items())):
            plt.sca(axes[i] if len(values) > 1 else axes)
            if len(v.shape) == 1:
                plt.bar(range(len(v)), v)
            else:
                plt.colorbar(plt.pcolormesh(v))
                plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
            plt.ylabel(" ".join(map(str, n)))
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        output_file = basename + ".png"
        plt.savefig(output_file)
        plt.clf()
        print("Saved '%s'." % output_file)


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("data", nargs="+")
    argparser.add_argument("-s", "--smoothing", type=int, default=100)
    main(argparser.parse_args())
