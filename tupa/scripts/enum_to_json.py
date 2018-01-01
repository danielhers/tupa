from configargparse import ArgParser

from tupa.model_util import save_json
from tupa.scripts.export import load_model


def main():
    argparser = ArgParser(description="Load TUPA model and save the features enumeration as a text JSON file.")
    argparser.add_argument("models", nargs="+", help="model file basename(s) to load")
    argparser.add_argument("-s", "--suffix", default=".enum.json", help="filename suffix to append")
    args = argparser.parse_args()
    for filename in args.models:
        model = load_model(filename)
        save_json(model.filename + args.suffix, model.feature_extractor.params)


if __name__ == "__main__":
    main()
