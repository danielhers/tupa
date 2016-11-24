import csv
import os
from collections import OrderedDict

import numpy as np

from parsing import parse, config
from parsing.config import Config
from parsing.w2v_util import load_word2vec
from ucca.evaluation import Scores


class Params(object):
    def __init__(self, params):
        self.params = params
        self.test_scores = None
        self.dev_scores = ()

    def run(self):
        assert Config().args.train and Config().args.passages or Config().args.folds, \
            "insufficient parameters given to parser"
        print("Running with %s" % self)
        Config().update(self.params)
        self.test_scores, self.dev_scores = parse.main()

    def score(self):
        return self.test_scores.average_f1() if self.test_scores is not None else \
            self.dev_scores[-1] if self.dev_scores else -float("inf")

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
    Config().args.nowrite = True
    out_file = os.environ.get("PARAMS_FILE", "params.csv")
    w2v_files = [os.environ[f] for f in os.environ if f.startswith("W2V_FILE")]
    num = int(os.environ.get("PARAMS_NUM", 30))
    np.random.seed()
    domains = (
        ("seed",            2147483647),  # max value for int
        ("classifier",      (config.MLP_NN, config.BILSTM_NN)),
        ("wordvectors",     [None] + [load_word2vec(f) for f in w2v_files]),
        ("updatewordvectors", [True, False]),
        ("worddim",         [50, 100, 200, 300]),
        ("tagdim",          (5, 10, 20)),
        ("labeldim",        (5, 10, 20)),
        ("punctdim",        (1, 2, 3)),
        ("gapdim",          (1, 2, 3)),
        ("actiondim",       (3, 5, 10)),
        ("layerdim",        (50, 100, 200, 300, 500, 1000)),
        ("layers",          [1] + 5 * [2]),
        ("lstmlayerdim",    (50, 100, 200, 300, 500, 1000)),
        ("lstmlayers",      [1] + 5 * [2]),
        ("activation",      config.ACTIVATIONS),
        ("init",            5 * [config.INITIALIZATIONS[0]] + list(config.INITIALIZATIONS)),
        ("batchsize",       (10, 30, 50, 100, 200, 500)),
        ("minibatchsize",   (50, 100, 200, 300, 500, 1000)),
        ("optimizer",       5 * [config.OPTIMIZERS[0]] + list(config.OPTIMIZERS)),
        ("importance",      (1, 2)),
        ("earlyupdate",     6 * [False] + [True]),
        ("iterations",      range(1, 31)),
        ("worddropout",     (0, .1, .2, .25, .3)),
        ("worddropoutexternal", (0, .1, .2, .25, .3)),
        ("dynet_l2",        (1e-7, 1e-6, 1e-5, 1e-4)),
        ("dropout",         (0, .1, .2, .3, .4, .5)),
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
