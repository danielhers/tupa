from collections import OrderedDict

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from configargparse import ArgParser
from scipy.misc import imresize
from matplotlib.ticker import MaxNLocator
from tupa.scripts.export import load_model
from tqdm import tqdm
import re

np.seterr("raise")

REPLACEMENTS = (
    ("axes_", ""),
    ("_", " "),
)


def smooth(x, s=100000):
    if len(x.shape) == 1:
        x = x.reshape((x.shape[0], -1))
    return imresize(x, size=(min(x.shape[0], s), min(x.shape[1], s)))


def visualize(model, filename):
    values = model.all_params()
    cols = OrderedDict()
    for key, value in values.items():
        if isinstance(value, np.ndarray):  # TODO group by prefix, showing W and b side by side
            cols.setdefault(key[:-2] + key[-1] if len(key) > 1 and key[-2] in "bW" else key, []).append((key, value))
    _, axes = plt.subplots(2, len(cols), figsize=(5*len(cols), 10))  # TODO https://stackoverflow.com/a/13784887/223267
    plt.tight_layout()
    for j, col in enumerate(tqdm(cols.values(), unit="param", desc=filename)):
        for i in range(2):
            axis = axes[i, j] if len(values) > 1 else axes
            if len(col) <= i:
                plt.delaxes(axis)
            else:
                plt.sca(axis)
                key, value = col[i]
                plt.colorbar(plt.pcolormesh(smooth(value)))
                for pattern, repl in REPLACEMENTS:  # TODO map 0123->ifoc
                    key = re.sub(pattern, repl, key)
                plt.title(key)
                for axis in (plt.gca().xaxis, plt.gca().yaxis):
                    axis.set_major_locator(MaxNLocator(integer=True))
    output_file = filename + ".png"
    plt.savefig(output_file)
    plt.clf()
    print("Saved '%s'." % output_file)


def main():
    argparser = ArgParser(description="Load TUPA model and visualize, saving to .png file.")
    argparser.add_argument("models", nargs="+", help="model file basename(s) to load")
    args = argparser.parse_args()
    for filename in args.models:
        model = load_model(filename)
        visualize(model, filename)


if __name__ == "__main__":
    main()
