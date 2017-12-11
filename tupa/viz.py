import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from argparse import ArgumentParser

np.seterr("raise")


def smooth(x, s=100):
    if len(x) < s:
        return x
    t = int(np.ceil(len(x) / s))
    s = int(len(x) / t)
    return [np.mean(x[i*t:(i+1)*t]) for i in range(s)]


def main(args):
    _, axes = plt.subplots(len(args.data), figsize=(19, 10))
    for i, filename in enumerate(args.data):
        print(filename)
        plt.sca(axes[i] if len(args.data) > 1 else axes)
        plt.xlabel(filename)
        end = 0
        ticks = []
        ticklabels = []
        name = None
        with open(filename) as f:
            for line in f:
                if line.startswith("#"):
                    name = line
                    print(name.strip())
                else:
                    values = smooth(np.fromstring(line, sep=" "), s=args.smoothing)
                    start = end
                    end += len(values)
                    ticks.append((start + end) / 2)
                    ticklabels.append(name.split()[1][1:6])
                    plt.bar(range(start, end), values)
        plt.xticks(ticks, ticklabels, rotation="vertical", fontsize=8)
        print()
    plt.savefig(args.output_file)
    print("Saved '%s'." % args.output_file)


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("data", nargs="+")
    argparser.add_argument("-o", "--output-file", default="viz.png")
    argparser.add_argument("-s", "--smoothing", type=int, default=100)
    main(argparser.parse_args())
