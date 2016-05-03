import os

import numpy as np

from parsing import parse
from parsing.config import Config, OPTIMIZERS, OBJECTIVES
from ucca.evaluation import Scores


class Params(object):
    def __init__(self, **kwargs):
        self.params = kwargs
        self.scores = None

    def run(self):
        assert Config().args.train and Config().args.passages or Config().args.folds, \
            "insufficient parameters given to parser"
        print("Running with %s" % self)
        for name, value in self.params.items():
            setattr(Config().args, name, value)
        self.scores = parse.main()
        assert self.score is not None, "parser failed to produce score"

    def score(self):
        return -float("inf") if self.scores is None else self.scores.average_unlabeled_f1()

    def __str__(self):
        ret = ", ".join("%s: %s" % (name, value) for name, value in self.params.items())
        if self.scores is not None:
            ret += ", average unlabeled f1: %.3f" % self.score()
        return ret

    def print(self, file):
        print(", ".join(list(map(str, self.params.values())) + [str(self.score())] + self.scores.fields()),
              file=file)

    def print_title(self, file):
        print(", ".join(list(self.params) + ["average unlabeled f1"] + Scores.field_titles()),
              file=file)


def main():
    Config().args.nowrite = True
    out_file = os.environ.get("PARAMS_FILE", "params.csv")
    num = int(os.environ.get("PARAMS_NUM", 30))
    param_values = {
        # "learningrate": np.round(0.001 + np.random.exponential(0.8, (num, 1)), 3),
        # "decayfactor": np.round(0.001 + np.random.exponential(0.8, (num, 1)), 3),
        "classifier": num * ["nn"],
        "tagdim": np.random.choice((5, 10, 20), num),
        "labeldim": np.random.choice((5, 10, 20), num),
        "punctdim": np.random.choice((1, 2, 3), num),
        "gapdim": np.random.choice((1, 2, 3), num),
        "layerdim": np.random.choice((50, 100, 200, 300, 500, 1000), num),
        "minibatchsize": np.random.choice((50, 100, 200, 300, 500, 1000), num),
        "nbepochs": np.random.choice((5, 10, 20, 30, 50, 100), num),
        "optimizer": np.random.choice(OPTIMIZERS, num),
        "loss": np.random.choice(OBJECTIVES, num),
    }
    params = list(set(Params(**{name: values[i]
                                for name, values in param_values.items()})
                      for i in range(num)))
    print("\n".join(["All parameter combinations to try: "] +
                    [str(h) for h in params]))
    print("Saving results to '%s'" % out_file)
    with open(out_file, "w") as f:
        params[0].print_title(f)
    for param in params:
        param.run()
        with open(out_file, "a") as f:
            param.print(f)
        best = max(params, key=Params.score)
        print("Best parameters: %s" % best)


if __name__ == "__main__":
    main()
