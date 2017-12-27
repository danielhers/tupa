import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from configargparse import ArgParser
from scipy.misc import imresize
from matplotlib.ticker import MaxNLocator
from tupa.export import load_model
from tqdm import tqdm

np.seterr("raise")

REPLACEMENTS = (
    ("axes_", ""),
    ("_", " "),
)


def smooth(x, s):
    if len(x.shape) == 1:
        x = x.reshape((-1, x.shape[0]))
    return imresize(x, size=(min(x.shape[0], s), min(x.shape[1], s)))


def visualize(model, filename, smoothing):
    values = model.get_all_params()
    for key, value in dict(values).items():
        if not isinstance(value, np.ndarray):
            del values[key]
    _, axes = plt.subplots(len(values), figsize=(19, 2 * len(values)))
    plt.tight_layout()
    for i, (key, value) in enumerate(tqdm(values.items(), unit="param", desc=filename)):
        plt.sca(axes[i] if len(values) > 1 else axes)
        plt.colorbar(plt.pcolormesh(smooth(value, smoothing)))
        for find, replace in REPLACEMENTS:
            key = key.replace(find, replace)
        plt.ylabel(key)
        for axis in (plt.gca().xaxis, plt.gca().yaxis):
            axis.set_major_locator(MaxNLocator(integer=True))
    output_file = filename + ".png"
    plt.savefig(output_file)
    plt.clf()
    print("Saved '%s'." % output_file)


def main():
    argparser = ArgParser(description="Load TUPA model and visualize, saving to .png file.")
    argparser.add_argument("models", nargs="+", help="model file basename(s) to load")
    argparser.add_argument("-s", "--smoothing", type=int, default=100)
    args = argparser.parse_args()
    for filename in args.models:
        model = load_model(filename)
        visualize(model, filename, args.smoothing)


if __name__ == "__main__":
    main()
