from ucca import evaluation
from ucca.constructions import PRIMARY

from scheme.conversion.sdp import SdpConverter

EVAL_TYPES = (evaluation.LABELED, evaluation.UNLABELED)


def get_scores(s1, s2, eval_type, verbose):
    converter = SdpConverter()
    edges = [{e for nodes, _ in converter.build_nodes(s) for n in nodes for e in n.outgoing} for s in (s1, s2)]
    if eval_type == evaluation.UNLABELED:
        for es in edges:
            for e in es:
                e.rel = None
    res = evaluation.EvaluatorResults({PRIMARY: evaluation.SummaryStatistics(
        len(edges[0] & edges[1]), len(edges[0] - edges[1]), len(edges[1] - edges[0]))}, default={PRIMARY.name: PRIMARY})
    if verbose:
        print("Evaluation type: (" + eval_type + ")")
        res.print()
    return res


def evaluate(guessed, ref, converter=None, verbose=False, **kwargs):
    del kwargs
    if converter is not None:
        guessed = converter(guessed)
        ref = converter(ref)
    return SdpScores((eval_type, get_scores(guessed, ref, eval_type, verbose)) for eval_type in EVAL_TYPES)


class SdpScores(evaluation.Scores):
    def __init__(self, *args, **kwargs):
        super(SdpScores, self).__init__(*args, **kwargs)
