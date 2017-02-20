import importlib.util  # needed for amr.peg
import os


def parse(*args, **kwargs):
    if parse.AMR is None:
        prev_dir = os.getcwd()
        os.chdir(os.path.dirname(importlib.util.find_spec("src.amr").origin))
        from src.amr import AMR
        os.chdir(prev_dir)
        parse.AMR = AMR
    return parse.AMR(*args, **kwargs)
parse.AMR = None


def evaluate(guessed, ref, amr_id=None, verbose=False):
    """
    Compare two AMRs and return scores, possibly printing them too.
    :param guessed: AMR object to evaluate
    :param ref: reference AMR object to compare to
    :param amr_id: ID of AMR pair
    :param verbose: whether to print the results
    :return: Scores object
    """
    if evaluate.smatch is None:
        import smatch
        evaluate.smatch = smatch
    evaluate.smatch.verbose = verbose
    return Scores(smatch.process_amr_pair([evaluate.smatch.get_amr_line(str(a)) for a in (guessed, ref)] + [amr_id]))
evaluate.smatch = None


class Scores(object):
    def __init__(self, counts):
        self.counts = counts
        self.precision, self.recall, self.f1 = evaluate.smatch.compute_f(*counts)

    @staticmethod
    def aggregate(scores):
        """
        Aggregate multiple Scores instances
        :param scores: iterable of Scores
        :return: new Scores with aggregated scores
        """
        return Scores(map(sum, zip(*[s.counts for s in scores])))

    def print(self):
        print(",".join(self.fields()))

    def fields(self):
        return self.precision, self.recall, self.f1

    @staticmethod
    def field_titles():
        return "precision", "recall", "f1"
