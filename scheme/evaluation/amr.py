import sys

from ucca import evaluation
from ucca.constructions import PRIMARY

from ..util.amr import *

sys.path.insert(0, os.path.dirname(importlib.util.find_spec("smatch.smatch").origin))  # to find amr.py from smatch
from smatch import smatch
sys.path.pop(0)

EVAL_TYPES = (evaluation.LABELED, evaluation.UNLABELED)

def get_scores(a1, a2, amr_id, eval_type, verbose):
    if eval_type == evaluation.UNLABELED:
        a1, a2 = [re.sub(":[a-zA-Z0-9-]*", ":label", a) for a in (a1, a2)]
    try:
        counts = smatch.process_amr_pair((a1, a2, amr_id))
    except (AttributeError, IndexError):  # error in one of the AMRs
        try:
            counts = smatch.process_amr_pair((a2, a2, amr_id))
            counts = (0, 0, counts[-1])  # best_match_num, test_triple_num
        except (AttributeError, IndexError):  # error in ref AMR
            counts = (0, 0, 1)  # best_match_num, test_triple_num, gold_triple_num
    res = SmatchResults(*counts)
    if verbose:
        print("Evaluation type: (" + eval_type + ")")
        res.print()
    return res


def evaluate(guessed, ref, converter=None, verbose=False, amr_id=None, **kwargs):
    """
    Compare two AMRs and return scores, possibly printing them too.
    :param guessed: AMR object to evaluate
    :param ref: reference AMR object to compare to
    :param converter: optional function to apply to inputs before evaluation
    :param amr_id: ID of AMR pair
    :param verbose: whether to print the results
    :return: SmatchScores object
    """
    del kwargs
    smatch.verbose = verbose
    a1, a2 = [read_amr(a, converter) for a in (guessed, ref)]
    return SmatchScores((eval_type, get_scores(a1, a2, amr_id, eval_type, verbose)) for eval_type in EVAL_TYPES)


def read_amr(amr, converter=None):
    if converter is not None:
        amr = converter(amr)
    return "".join(l for l in (amr if isinstance(amr, list) else str(amr).splitlines()) if not l.startswith("#"))


class SmatchResults(evaluation.EvaluatorResults):
    def __init__(self, *counts):
        super().__init__({PRIMARY: evaluation.SummaryStatistics(*counts)}, default={PRIMARY.name: PRIMARY})
        self.p, self.r, self.f1 = smatch.compute_f(*counts)


class SmatchScores(evaluation.Scores):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def name():
        return "AMR"

    @staticmethod
    def format():
        return "amr"

    def __str__(self):
        return ",".join(self.fields())
