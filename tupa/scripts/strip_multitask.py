import os
from configargparse import ArgParser

from tupa.config import FORMATS
from tupa.model import NODE_LABEL_KEY
from tupa.scripts.export import load_model


def strip_multitask(model, keep):
    if "amr" not in keep:  # Remove AMR-specific features: node label and category
        delete_if_exists((model.feature_params, model.classifier.params), (NODE_LABEL_KEY, "c"))
    delete_if_exists((model.classifier.labels, model.classifier.axes), {NODE_LABEL_KEY}.union(FORMATS).difference(keep))


def delete_if_exists(dicts, keys):
    for d in dicts:
        for key in keys:
            try:
                del d[key]
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
    argparser.add_argument("-k", "--keep", nargs="+", choices=tuple(filter(None, FORMATS)), default=["ucca"],
                           help="tasks to keep features/weights for")
    argparser.add_argument("-o", "--out-dir", default=".", help="directory to write modified model files to")
    main(argparser.parse_args())
