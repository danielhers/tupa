import importlib.util  # needed for amr.peg
import os
import sys

from tupa import constraints

sys.path.insert(0, os.path.dirname(importlib.util.find_spec("smatch.smatch").origin))  # to find amr.py from smatch
from smatch import smatch
sys.path.pop(0)

prev_dir = os.getcwd()
os.chdir(os.path.dirname(importlib.util.find_spec("src.amr").origin))  # to find amr.peg
try:
    from src import amr as amr_lib
finally:
    os.chdir(prev_dir)

NODE_LABEL_ATTRIB = "label"


def parse(*args, **kwargs):
    return amr_lib.AMR(*args, **kwargs)


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
        return "".join(str(amr).splitlines())
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
    return Scores(counts)


class Scores(object):
    def __init__(self, counts):
        self.counts = counts
        self.precision, self.recall, self.f1 = smatch.compute_f(*counts)

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
        return Scores(map(sum, zip(*[s.counts for s in scores])))

    def print(self, *args, **kwargs):
        print("Precision: %.3f\nRecall: %.3f\nF1: %.3f" % (self.precision, self.recall, self.f1), *args, **kwargs)

    def fields(self):
        return ["%.3f" % float(f) for f in (self.precision, self.recall, self.f1)]

    def titles(self):
        return self.field_titles()

    @staticmethod
    def field_titles(*args, **kwargs):
        del args, kwargs
        return ["precision", "recall", "f1"]

    def __str__(self):
        print(",".join(self.fields()))


class Constraints(constraints.Constraints):
    def __init__(self, args):
        super(Constraints, self).__init__(args)

    require_connected = True
    require_first_shift = False
    require_implicit_childless = False
    allow_root_terminal_children = True
    allow_multiple_edges = True

    UniqueOutgoing = {
        "instance-of",
        "ARG",
        "op",
        "snt",
    }

    def is_unique_outgoing(self, tag):
        return tag.rstrip("0123456789") in self.UniqueOutgoing

    is_unique_incoming = None
    mutually_exclusive_outgoing = None
    childless_incoming = None
    childless_outgoing = None
    is_scene_sufficient_outgoing = None
    is_scene_necessary_outgoing = None
    is_scene_sufficient_incoming = None
    is_top_level = None
    is_linker_incoming = None

    def is_possible_multiple_incoming(self, tag):
        return False
