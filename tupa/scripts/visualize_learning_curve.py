import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from configargparse import ArgParser
import os
from glob import glob

np.seterr("raise")

PRIMARY_F1_COLUMN = 3
REMOTE_F1_COLUMN = 6


def load_scores(basename, div="dev"):
    filename = "%s.%s.csv" % (basename, div)
    print("Loading %s scores from '%s'" % (div, filename))
    offset = 0
    try:
        with open(filename) as f:
            if "iteration" not in f.readline():
                offset = -1
            scores = np.genfromtxt(f, delimiter=",", invalid_raise=False)
    except ValueError as e:
        raise ValueError("Failed reading '%s'" % filename) from e
    try:
        return scores[:, [PRIMARY_F1_COLUMN + offset, REMOTE_F1_COLUMN + offset]]
    except IndexError:
        try:
            return scores[:, PRIMARY_F1_COLUMN + offset]
        except IndexError as e:
            raise ValueError("Failed reading '%s'" % filename) from e


def visualize(scores, filename, div="dev"):
    plt.plot(range(1, 1 + len(scores)), scores)
    plt.xlabel("epochs")
    plt.ylabel("%s f1" % div)
    if len(scores.shape) > 1:
        plt.legend(["primary", "remote"])
    plt.title(filename)
    output_file = "%s.%s.png" % (filename, div)
    plt.savefig(output_file)
    plt.clf()
    print("Saved '%s'." % output_file)


def main():
    argparser = ArgParser(description="Visualize scores of a model over the dev set, saving to .png file.")
    argparser.add_argument("models", nargs="+", help="model file basename(s) to load")
    args = argparser.parse_args()
    for pattern in args.models:
        for filename in sorted(glob(pattern)) or [pattern]:
            basename, _ = os.path.splitext(filename)
            for div in "dev", "test":
                try:
                    scores = load_scores(basename, div=div)
                except OSError:
                    continue
                visualize(scores, basename, div=div)


if __name__ == "__main__":
    main()
