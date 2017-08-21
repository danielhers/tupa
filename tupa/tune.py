import csv

import numpy as np
import os
from collections import OrderedDict

from tupa import parse, config
from tupa.config import Config

MODELS_DIR = "models"


class Params(object):
    def __init__(self, params):
        self.params = params
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


def get_values_based_on_format(*values):
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
        ("learning_rate",           (0.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5)),
        ("learning_rate_decay",     (0.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5)),
        ("update_word_vectors",     [True, False]),
        ("word_vectors",            [None] + word_vectors_files),
        ("word_dim_external",       (0, 300)),
        ("word_dim",                (0, 50, 100, 200, 300)),
        ("tag_dim",                 (5, 10, 20)),
        ("dep_dim",                 (5, 10, 20)),
        ("edge_label_dim",          (5, 10, 20)),
        ("node_label_dim",          get_values_based_on_format(10, 20, 30)),
        ("node_category_dim",       get_values_based_on_format(3, 5, 10)),
        ("max_node_categories",     get_values_based_on_format(10, 25)),
        ("punct_dim",               (1, 2, 3)),
        ("action_dim",              (3, 5, 10)),
        ("ner_dim",                 (3, 5, 10)),
        ("max_node_labels",         get_values_based_on_format(500, 750, 1000, 1500, 2000)),
        ("min_node_label_count",    range(1, 201)),
        ("layer_dim",               (50, 100, 200, 300, 500, 1000)),
        ("layers",                  [1] + 5 * [2]),
        ("lstm_layer_dim",          (50, 100, 200, 300, 500, 1000)),
        ("lstm_layers",             [1] + 5 * [2]),
        ("embedding_layer_dim",     (50, 100, 200, 300, 500, 1000)),
        ("embedding_layers",        5 * [1] + [2]),
        ("output_dim",              (20, 30, 50, 100, 200, 300, 500, 1000)),
        ("activation",              config.ACTIVATIONS),
        ("init",                    5 * [config.INITIALIZATIONS[0]] + list(config.INITIALIZATIONS)),
        ("batch_size",              (10, 30, 50, 100, 200, 500)),
        ("minibatch_size",          (50, 100, 200, 300, 500, 1000)),
        ("optimizer",               config.OPTIMIZERS),
        ("swap_importance",         (1, 2)),
        ("iterations",              range(1, 51)),
        ("word_dropout",            (0, .1, .2, .25, .3)),
        ("word_dropout_external",   (0, .1, .2, .25, .3)),
        ("dynet_weight_decay",      (1e-7, 1e-6, 1e-5, 1e-4)),
        ("dropout",                 (0, .1, .2, .3, .4, .5)),
        ("require_connected",       [True, False]),
        ("swap",                    [config.REGULAR, config.COMPOUND]),
        ("max_swap",                range(2, 21)),
        ("max_words",               (2000, 5000, 7500, 10000, 20000)),
        ("max_words_external",      (None, 5000, 10000, 30000)),
    )
    params = [Params(OrderedDict(p))
              for p in zip(*[[(n, v.item() if hasattr(v, "item") else v)
                              for v in np.random.choice(vs, num)]
                             for n, vs in domains])]
    print("All parameter combinations to try:")
    print("\n".join(map(str, params)))
    print("Saving results to '%s'" % out_file)
    for param in params:
        param.run(out_file)
        best = max(params, key=Params.score)
        print("Best parameters: %s" % best)


if __name__ == "__main__":
    main()
