import os
from configargparse import ArgParser

from tupa.model import NODE_LABEL_KEY
from tupa.scripts.export import load_model


def strip_multitask(model, keep):
    if "amr" not in keep:
        for d in (model.feature_params, model.classifier.params):
            for key in (NODE_LABEL_KEY, "c"):
                try:
                    del d[key]
                except KeyError:
                    pass
    for d in (model.classifier.labels, model.classifier.axes):
        for axis in set(d).difference(keep):
            try:
                del d[axis]
            except KeyError:
                pass


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    for filename in args.models:
        model = load_model(filename)
        strip_multitask(model, args.keep)
        model.filename = os.path.join(args.out_dir, os.path.basename(filename))
        model.save()


if __name__ == "__main__":
    argparser = ArgParser(description="Load TUPA model and save with just one task's features/weights.")
    argparser.add_argument("models", nargs="+", help="model file basename(s) to load")
    argparser.add_argument("-k", "--keep", nargs="+", default=["ucca"], help="tasks to keep features/weights for")
    argparser.add_argument("-o", "--out-dir", default=".", help="filename suffix to append")
    main(argparser.parse_args())
