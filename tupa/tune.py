import csv
import os
from collections import OrderedDict

import numpy as np

from tupa import parse, config
from tupa.config import Config

MODELS_DIR = "models"


class Params(object):
    def __init__(self, params):
        self.params = params
        if self.params["classifier"] != config.BILSTM_NN:
            self.params["rnn"] = None
            self.params["lstm_layer_dim"] = self.params["lstm_layers"] = \
                self.params["embedding_layer_dim"] = self.params["embedding_layers"] = 0
        if self.params["swap"] != config.COMPOUND:
            self.params["max_swap"] = 1
        if not self.params["word_dim_external"]:
            self.params["max_words_external"] = self.params["word_dropout_external"] = 0
            self.params["word_vectors"] = self.params["update_word_vectors"] = None
        if not self.params["word_dim"]:
            self.params["max_words"] = self.params["word_dropout"] = 0
        self.params["model"] = "%s/%s-%d" % (MODELS_DIR, self.params["classifier"], self.params["seed"])
        self.scores = None

    def run(self, out_file):
        assert Config().args.train and (Config().args.passages or Config().args.dev) or \
               Config().args.passages and Config().args.folds, "insufficient parameters given to parser"
        print("Running with %s" % self)
        Config().update(self.params)
        for i, self.scores in enumerate(parse.main()):
            print_title = not os.path.exists(out_file)
            with open(out_file, "a") as f:
                if print_title:
                    csv.writer(f).writerow([p for p in self.params.keys()] +
                                           ["average_labeled_f1"] + self.scores.titles())
                csv.writer(f).writerow([str((i + 1) if n == "iterations" else p) for n, p in self.params.items()] +
                                       [str(self.scores.average_f1())] + self.scores.fields())

    def score(self):
        return -float("inf") if self.scores is None else self.scores.average_f1()

    def __str__(self):
        ret = ", ".join("%s: %s" % (name, value) for name, value in self.params.items())
        if self.scores is not None:
            ret += ", average labeled f1: %.3f" % self.score()
        return ret


def get_values_based_on_format(values):
    return values if Config().args.format == "amr" else (0,)


def main():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    Config().args.write = False
    out_file = os.environ.get("PARAMS_FILE", "params.csv")
    word_vectors_files = [os.environ[f] for f in os.environ if f.startswith("WORD_VECTORS")]
    num = int(os.environ.get("PARAMS_NUM", 30))
    np.random.seed()
    domains = (
        ("seed",                    2147483647),  # max value for int
        ("classifier",              (config.MLP_NN, config.BILSTM_NN)),
        ("learning_rate",           np.logspace(-5, 0, 11)),
        ("learning_rate_decay",     np.r_[0, np.logspace(-5, -1, 9)]),
        ("update_word_vectors",     [True, False]),
        ("word_vectors",            [None] + word_vectors_files),
        ("word_dim_external",       (0, 300)),
        ("word_dim",                range(0, 301)),
        ("tag_dim",                 range(0, 21)),
        ("dep_dim",                 range(0, 21)),
        ("edge_label_dim",          range(0, 21)),
        ("node_label_dim",          get_values_based_on_format(range(30))),
        ("node_category_dim",       get_values_based_on_format(range(15))),
        ("max_node_categories",     get_values_based_on_format(range(10, 26))),
        ("punct_dim",               range(4)),
        ("action_dim",              range(15)),
        ("ner_dim",                 range(15)),
        ("max_node_labels",         get_values_based_on_format(range(500, 2001))),
        ("min_node_label_count",    range(1, 201)),
        ("layer_dim",               range(50, 1001)),
        ("layers",                  range(1, 4)),
        ("lstm_layer_dim",          range(50, 1001, 2)),
        ("lstm_layers",             range(1, 4)),
        ("embedding_layer_dim",     range(50, 1001)),
        ("embedding_layers",        range(1, 4)),
        ("output_dim",              range(20, 1001)),
        ("activation",              config.ACTIVATIONS),
        ("init",                    5 * [config.INITIALIZATIONS[0]] + list(config.INITIALIZATIONS)),
        ("batch_size",              range(10, 501)),
        ("minibatch_size",          range(50, 1001)),
        ("optimizer",               config.OPTIMIZERS),
        ("swap_importance",         (1, 2)),
        ("iterations",              range(1, 51)),
        ("word_dropout",            np.arange(.31, step=.01)),
        ("word_dropout_external",   np.arange(.31, step=.01)),
        ("dynet_weight_decay",      np.logspace(-7, -4, 7)),
        ("dropout",                 np.arange(.51, step=.01)),
        ("require_connected",       [True, False]),
        ("swap",                    [config.REGULAR, config.COMPOUND]),
        ("max_swap",                range(2, 21)),
        ("max_words",               range(2000, 20001)),
        ("max_words_external",      [None] + list(range(5000, 30001))),
        ("rnn",                     config.RNNS),
    )
    params = [Params(OrderedDict(p)) for p in zip(*[[(n, v.item() if hasattr(v, "item") else v)
                                                     for v in np.random.choice(vs, num)] for n, vs in domains])]
    print("All parameter combinations to try:")
    print("\n".join(map(str, params)))
    print("Saving results to '%s'" % out_file)
    for param in params:
        param.run(out_file)
        best = max(params, key=Params.score)
        print("Best parameters: %s" % best)


if __name__ == "__main__":
    main()
