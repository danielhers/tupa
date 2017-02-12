import csv
import os
from collections import OrderedDict

import numpy as np

from parsing import parse, config
from parsing.config import Config
from ucca.evaluation import Scores

MODELS_DIR = "models"


class Params(object):
    def __init__(self, params):
        if params["classifier"] == config.FEEDFORWARD_NN:
            params["iterations"] = 1
        self.params = params
        self.params["model"] = "%s/ucca-%s-%d" % (MODELS_DIR, self.params["classifier"], self.params["seed"])
        self.test_scores = None
        self.dev_scores = ()

    def run(self):
        assert Config().args.train and (Config().args.passages or Config().args.dev) or \
               Config().args.passages and Config().args.folds, "insufficient parameters given to parser"
        print("Running with %s" % self)
        Config().update(self.params)
        self.test_scores, self.dev_scores = parse.main()

    def score(self):
        return self.test_scores.average_f1() if self.test_scores is not None else \
            self.dev_scores[-1].average_f1() if self.dev_scores else -float("inf")

    def __str__(self):
        ret = ", ".join("%s: %s" % (name, value) for name, value in self.params.items())
        if self.test_scores is not None:
            ret += ", average labeled f1: %.3f" % self.score()
        return ret

    def get_fields(self):
        return [str(p) for p in self.params.values()] + [str(self.score())] + self.test_scores.fields()

    def get_field_titles(self):
        return [p for p in self.params.keys()] + ["average_labeled_f1"] + Scores.field_titles()

    def write_scores(self, writer):
        for i, scores in enumerate(self.dev_scores):
            writer.writerow([str((i + 1) if n == "iterations" else p) for n, p in self.params.items()] +
                            [str(scores.average_f1())] + scores.fields())
        if self.test_scores is not None and (not self.dev_scores or
                                             self.test_scores.fields() != self.dev_scores[-1].fields()):
            writer.writerow([str(p) for p in self.params.values()] +
                            [str(self.test_scores.average_f1())] + self.test_scores.fields())


def main():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    Config().args.no_write = True
    out_file = os.environ.get("PARAMS_FILE", "params.csv")
    word_vectors_files = [os.environ[f] for f in os.environ if f.startswith("WORD_VECTORS")]
    num = int(os.environ.get("PARAMS_NUM", 30))
    np.random.seed()
    domains = (
        ("seed",                    2147483647),  # max value for int
        ("classifier",              (config.SPARSE_PERCEPTRON,)),
#        ("word_vectors",            [None] + word_vectors_files),
#        ("word_dim_external",       (0, 300)),
#        ("word_dim",                (0, 50, 100, 200, 300)),
#        ("tag_dim",                 (5, 10, 20)),
#        ("dep_dim",                 (5, 10, 20)),
#        ("label_dim",               (5, 10, 20)),
#        ("punct_dim",               (1, 2, 3)),
#        ("gap_dim",                 (1, 2, 3)),
#        ("action_dim",              (3, 5, 10)),
        ("batch_size",              (10, 30, 50, 100, 200, 500)),
#        ("minibatch_size",          (50, 100, 200, 300, 500, 1000)),
        ("swap_importance",         (1, 2)),
        ("iterations",              range(1, 51)),
        ("min_update",              range(1, 51)),
        ("learning_rate",           (.1, .5, .9, 1, 1.1, 1.5, 2)),
        ("learning_rate_decay",     (.1, .5, .8, .9, .99, .999, .9999)),
#        ("word_dropout",            (0, .1, .2, .25, .3)),
#        ("word_dropout_external",   (0, .1, .2, .25, .3)),
    )
    params = [Params(OrderedDict(p))
              for p in zip(*[[(n, v.item() if hasattr(v, "item") else v)
                              for v in np.random.choice(vs, num)]
                             for n, vs in domains])]
    print("All parameter combinations to try:")
    print("\n".join(map(str, params)))
    print("Saving results to '%s'" % out_file)
    with open(out_file, "w") as f:
        csv.writer(f).writerow(params[0].get_field_titles())
    for param in params:
        param.run()
        with open(out_file, "a") as f:
            param.write_scores(csv.writer(f))
        best = max(params, key=Params.score)
        print("Best parameters: %s" % best)


if __name__ == "__main__":
    main()
