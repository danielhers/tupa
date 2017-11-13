from argparse import ArgumentParser

import numpy as np

from tupa.model import Model


def export_model(filename):
    model = Model(model_type=None, filename=filename)
    model.load()
    out_file = filename + ".npz"
    np.savez_compressed(out_file, **model.classifier.get_all_params())
    print("Wrote '%s'" % out_file)


def main():
    argparser = ArgumentParser(description="Load TUPA model and export as .npz file.")
    argparser.add_argument("model", help="model file basename to load")
    args = argparser.parse_args()
    export_model(args.model)


if __name__ == "__main__":
    main()
