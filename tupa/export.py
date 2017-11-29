import numpy as np
from configargparse import ArgParser

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


def main():
    argparser = ArgParser(description="Load TUPA model and export as .npz file.")
    argparser.add_argument("model", help="model file basename to load")
    args = argparser.parse_args()
    model = load_model(args.model)
    save_model(model, args.model)
    Config().save(args.model)


if __name__ == "__main__":
    main()
