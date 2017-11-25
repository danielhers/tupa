from configargparse import ArgParser

import numpy as np
import shlex

from tupa.config import Config
from tupa.model import Model


def load_model(filename):
    model = Model(model_type=None, filename=filename)
    model.load()
    return model


def save_model(model, filename):
    out_file = filename + ".npz"
    np.savez_compressed(out_file, **model.get_all_params())
    print("Wrote '%s'" % out_file)


def save_config(filename):
    out_file = filename + ".yml"
    with open(out_file, "w") as f:
        name = None
        values = []
        for arg in shlex.split(str(Config()), "--") + ["--"]:
            if arg.startswith("--"):
                if name:
                    if len(values) > 1:
                        values[0] = "[" + values[0]
                        values[-1] += "]"
                    print("%s: %s" % (name, ", ".join(values) or "true"), file=f)
                name = arg[2:]
                values = []
            else:
                values.append(arg)
    print("Wrote '%s'" % out_file)


def main():
    argparser = ArgParser(description="Load TUPA model and export as .npz file.")
    argparser.add_argument("model", help="model file basename to load")
    args = argparser.parse_args()
    model = load_model(args.model)
    save_model(model, args.model)
    save_config(args.model)


if __name__ == "__main__":
    main()
