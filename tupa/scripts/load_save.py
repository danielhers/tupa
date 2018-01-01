from configargparse import ArgParser

from tupa.scripts.export import load_model


def main():
    argparser = ArgParser(description="Load TUPA model and save again to a different file.")
    argparser.add_argument("models", nargs="+", help="model file basename(s) to load")
    argparser.add_argument("-s", "--suffix", default=".1", help="filename suffix to append")
    args = argparser.parse_args()
    for filename in args.models:
        model = load_model(filename)
        model.filename += args.suffix
        model.classifier.filename += args.suffix
        model.save()


if __name__ == "__main__":
    main()
