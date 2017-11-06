import sys

from ..util.amr import *

sys.path.insert(0, os.path.dirname(importlib.util.find_spec("smatch.smatch").origin))  # to find amr.py from smatch
from smatch import smatch
sys.path.pop(0)


def evaluate(guessed, ref, converter=None, verbose=False, amr_id=None, **kwargs):
    """
    Compare two AMRs and return scores, possibly printing them too.
    :param guessed: AMR object to evaluate
    :param ref: reference AMR object to compare to
    :param converter: optional function to apply to inputs before evaluation
    :param amr_id: ID of AMR pair
    :param verbose: whether to print the results
    :return: Scores object
    """
    def _read_amr(amr):
        return "".join(l for l in (amr if isinstance(amr, list) else str(amr).splitlines()) if not l.startswith("#"))
    del kwargs
    if converter is not None:
        guessed = converter(guessed)
        ref = converter(ref)
    smatch.verbose = verbose
    guessed = _read_amr(guessed)
    ref = _read_amr(ref)
    try:
        counts = smatch.process_amr_pair((guessed, ref, amr_id))
    except (AttributeError, IndexError):  # error in one of the AMRs
        try:
            counts = smatch.process_amr_pair((ref, ref, amr_id))
            counts = (0, 0, counts[-1])  # best_match_num, test_triple_num
        except (AttributeError, IndexError):  # error in ref AMR
            counts = (0, 0, 1)  # best_match_num, test_triple_num, gold_triple_num
    return SmatchScores(counts)


class SmatchScores(object):
    def __init__(self, counts):
        self.counts = counts
        self.precision, self.recall, self.f1 = smatch.compute_f(*counts)

    @staticmethod
    def name():
        return "AMR"

    def average_f1(self, *args, **kwargs):
        del args, kwargs
        return self.f1

    @staticmethod
    def aggregate(scores):
        """
        Aggregate multiple Scores instances
        :param scores: iterable of Scores
        :return: new Scores with aggregated scores
        """
        return SmatchScores(map(sum, zip(*[s.counts for s in scores])))

    def print(self, *args, **kwargs):
        print("Smatch precision: %.3f\nSmatch recall: %.3f\nSmatch F1: %.3f\n" % (self.precision, self.recall, self.f1),
              *args, **kwargs)

    def fields(self):
        return ["%.3f" % float(f) for f in (self.precision, self.recall, self.f1)]

    def titles(self):
        return self.field_titles()

    @staticmethod
    def field_titles(*args, **kwargs):
        del args, kwargs
        return ["precision", "recall", "f1"]

    def __str__(self):
        return ",".join(self.fields())
