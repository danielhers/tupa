from configargparse import ArgParser
from ucca.textutil import get_vocab

from tupa.model_util import save_json
from tupa.scripts.export import load_model


def decode(vocab, value):
    if isinstance(value, int):
        try:
            return vocab.strings[value]
        except KeyError:
            pass
    return value


def main():
    argparser = ArgParser(description="Load TUPA model and save the features enumeration as a text JSON file.")
    argparser.add_argument("models", nargs="+", help="model file basename(s) to load")
    argparser.add_argument("-s", "--suffix", default=".enum.json", help="filename suffix to append")
    argparser.add_argument("-l", "--lang", help="use spaCy model to decode numeric IDs")
    args = argparser.parse_args()
    for filename in args.models:
        model = load_model(filename)
        params = model.feature_extractor.params
        if args.lang:
            vocab = get_vocab(lang=args.lang)
            for param in params.values():
                if param.data:
                    param.data = [decode(vocab, v) for v in sorted(param.data, key=param.data.get)]
        save_json(model.filename + args.suffix, params)


if __name__ == "__main__":
    main()
