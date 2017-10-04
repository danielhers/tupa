import csv
import os
from collections import OrderedDict

import numpy as np

from tupa import parse, config
from tupa.config import Config

MODELS_DIR = "models"


class Params(object):
    def __init__(self, params, **hyperparams):
        self.params = params
        self.hyperparams = hyperparams
        if self.params["classifier"] != config.BILSTM_NN:
            for p in [self.params] + list(self.hyperparams.values()):
                p["rnn"] = None
                p["lstm_layer_dim"] = p["lstm_layers"] = p["embedding_layer_dim"] = p["embedding_layers"] = 0
        if self.params["swap"] != config.COMPOUND:
            self.params["max_swap"] = 1
        if not self.params["word_dim_external"]:
            self.params["max_words_external"] = self.params["word_dropout_external"] = 0
            self.params["word_vectors"] = self.params["update_word_vectors"] = None
        if not self.params["word_dim"]:
            self.params["max_words"] = self.params["word_dropout"] = 0
        self.params["model"] = "%s/%s-%d" % (MODELS_DIR, self.params["classifier"], self.params["seed"])
        self.scores = None
        self.all_params = OrderedDict(list(self.params.items()) +
                                      [(a + "." + k, v) for a, p in self.hyperparams.items() for k, v in p.items()])

    def run(self, out_file):
        assert Config().args.train and (Config().args.passages or Config().args.dev) or \
               Config().args.passages and Config().args.folds, "insufficient parameters given to parser"
        print("Running with %s" % self)
        Config().update(self.params)
        Config().update_hyperparams(**self.hyperparams)
        for i, self.scores in enumerate(parse.main(), start=1):
            print_title = not os.path.exists(out_file)
            with open(out_file, "a") as f:
                if print_title:
                    csv.writer(f).writerow([k for k in self.all_params.keys()] +
                                           ["average_labeled_f1"] + self.scores.titles())
                csv.writer(f).writerow([str(i if n == "iterations" else p) for n, p in self.all_params.items()] +
                                       [str(self.scores.average_f1())] + self.scores.fields())

    def score(self):
        return -float("inf") if self.scores is None else self.scores.average_f1()

    def __str__(self):
        ret = ", ".join("%s: %s" % (k, v) for k, v in self.all_params.items())
        if self.scores is not None:
            ret += ", average labeled f1: %.3f" % self.score()
        return ret


def get_values_based_on_format(values):
    return values if "amr" in (Config().args.formats or ()) else (0,)


def main():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    Config().args.write = False
    if not Config().args.verbose:
        Config().args.verbose = 1
    out_file = os.environ.get("PARAMS_FILE", "params.csv")
    word_vectors_files = [os.environ[f] for f in os.environ if f.startswith("WORD_VECTORS")]
    size = int(os.environ.get("PARAMS_NUM", 30))
    np.random.seed()
    domains = (
        # Parameter name            Shared  Domain of possible values
        ("seed",                    False,  2147483647),  # max value for int
        ("classifier",              False,  (config.MLP_NN, config.BILSTM_NN)),
        ("learning_rate",           False,  np.logspace(-5, 0, 11)),
        ("learning_rate_decay",     False,  np.r_[0, np.logspace(-5, -1, 9)]),
        ("update_word_vectors",     False,  [True, False]),
        ("word_vectors",            False,  [None] + word_vectors_files),
        ("word_dim_external",       False,  (0, 300)),
        ("word_dim",                False,  301),
        ("tag_dim",                 False,  21),
        ("dep_dim",                 False,  21),
        ("edge_label_dim",          False,  21),
        ("node_label_dim",          False,  get_values_based_on_format(range(30))),
        ("node_category_dim",       False,  get_values_based_on_format(range(15))),
        ("max_node_categories",     False,  get_values_based_on_format(range(10, 26))),
        ("punct_dim",               False,  range(4)),
        ("action_dim",              False,  range(15)),
        ("ner_dim",                 False,  range(15)),
        ("max_node_labels",         False,  get_values_based_on_format(range(1000, 4001))),
        ("min_node_label_count",    False,  range(1, 101)),
        ("layer_dim",               False,  range(50, 1001)),
        ("layers",                  False,  range(1, 3)),
        ("lstm_layer_dim",          True,   range(50, 501, 2)),
        ("lstm_layers",             True,   range(1, 3)),
        ("embedding_layer_dim",     True,   range(50, 501)),
        ("embedding_layers",        True,   range(1, 3)),
        ("output_dim",              False,  range(40, 351)),
        ("activation",              True,   ("cube", "tanh", "relu")),
        ("init",                    True,   ("glorot_uniform", "normal")),
        ("minibatch_size",          False,  range(50, 2001)),
        ("optimizer",               False,  list(config.TRAINERS)),
        ("swap_importance",         False,  np.arange(1, 2, step=.1)),
        ("iterations",              False,  range(1, 51)),
        ("word_dropout",            False,  np.arange(.31, step=.01)),
        ("word_dropout_external",   False,  np.arange(.51, step=.01)),
        ("tag_dropout",             False,  np.arange(.31, step=.01)),
        ("dep_dropout",             False,  np.arange(.31, step=.01)),
        ("node_label_dropout",      False,  np.arange(.31, step=.01)),
        ("dynet_weight_decay",      False,  np.logspace(-7, -4, 7)),
        ("dropout",                 False,  np.arange(.51, step=.01)),
        ("require_connected",       False,  [True, False]),
        ("swap",                    False,  [config.REGULAR, config.COMPOUND]),
        ("max_swap",                False,  range(2, 21)),
        ("max_words",               False,  range(2000, 20001)),
        ("max_words_external",      False,  [None] + list(range(5000, 100001))),
        ("rnn",                     True,   list(config.RNNS)),
    )
    params = [Params(p, shared=s) for p, s in zip(*[map(OrderedDict, zip(*[sample(name, domain, size)
                                                                           for name, shared, domain in domains
                                                                           if shared or all_parameters]))
                                                    for all_parameters in (True, False)])]
    print("All parameter combinations to try:")
    print("\n".join(map(str, params)))
    print("Saving results to '%s'" % out_file)
    for param in params:
        param.run(out_file)
        best = max(params, key=Params.score)
        print("Best parameters: %s" % best)


def sample(name, domain, size):
    return [(name, v.item() if hasattr(v, "item") else v) for v in np.random.choice(domain, size)]


if __name__ == "__main__":
    main()
