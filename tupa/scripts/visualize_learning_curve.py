import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from configargparse import ArgParser

np.seterr("raise")

PRIMARY_F1_COLUMN = 3
REMOTE_F1_COLUMN = 6


def load_scores(filename):
    print("Loading dev scores from '%s'" % filename)
    scores = np.loadtxt(filename + ".dev.csv", delimiter=",", skiprows=1)
    try:
        return scores[:, [PRIMARY_F1_COLUMN, REMOTE_F1_COLUMN]]
    except IndexError:
        return scores[:, PRIMARY_F1_COLUMN]


def visualize(scores, filename):
    plt.plot(range(1, 1 + len(scores)), scores)
    plt.xlabel("epochs")
    plt.ylabel("dev f1")
    plt.legend(["primary", "remote"])
    plt.title(filename)
    output_file = filename + ".dev.png"
    plt.savefig(output_file)
    plt.clf()
    print("Saved '%s'." % output_file)


def main():
    argparser = ArgParser(description="Visualize scores of a model over the dev set, saving to .png file.")
    argparser.add_argument("models", nargs="+", help="model file basename(s) to load")
    args = argparser.parse_args()
    for filename in args.models:
        scores = load_scores(filename)
        visualize(scores, filename)


if __name__ == "__main__":
    main()
