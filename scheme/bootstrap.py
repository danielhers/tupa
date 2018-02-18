#!/usr/bin/env python3

import os

import configargparse
import numpy as np
from tqdm import tqdm
from ucca import evaluation

from scheme.cfgutil import add_verbose_argument
from scheme.evaluate import EVALUATORS, passage_format, evaluate_all, Scores

desc = """Evaluates statistical significance of F1 scores between two systems."""


def main(args):
    files = [[os.path.join(d, f) for f in os.listdir(d)] for d in args.guessed + [args.ref]]
    n = len(files[-1])
    evaluate = EVALUATORS.get(passage_format(files[-1][0])[1], EVALUATORS[args.format]).evaluate
    results = [list(evaluate_all(args, evaluate, f, n)) for f, n in zip((files[0::2], files[1:]), args.guessed)]
    d = diff(results, verbose=True)
    sample = np.random.choice(n, (args.nboot, n))
    s = np.sum(np.sign(d) * diff(results, indices) > 2 * np.abs(d) for indices in tqdm(sample, unit=" samples"))
    print("p-value:")
    print(s / args.nboot)


def diff(results, indices=None, verbose=False):
    scores = [Scores(r if indices is None else [r[i] for i in indices]) for r in results]
    fields = np.array([s.fields() for s in scores], dtype=float)
    if verbose:
        print(" ".join(evaluation.Scores.field_titles()))
        print("\n".join(map(str, fields)))
    return fields[1] - fields[0]


if __name__ == '__main__':
    argparser = configargparse.ArgParser(description=desc)
    argparser.add_argument("guessed", nargs=2, help="directories for the guessed annotations: baseline, evaluated")
    argparser.add_argument("ref", help="directory for the reference annotations")
    argparser.add_argument("-b", "--nboot", type=int, default=int(1e4), help="number of bootstrap samples")
    argparser.add_argument("-f", "--format", default="amr", help="default format (if cannot determine by suffix)")
    group = argparser.add_mutually_exclusive_group()
    add_verbose_argument(group, help="detailed evaluation output")
    group.add_argument("-q", "--quiet", action="store_true", help="do not print anything")
    main(argparser.parse_args())
