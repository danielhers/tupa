import importlib.util  # needed for amr.peg
import os
import re
import sys

from nltk.corpus import wordnet as wn, propbank as pb

from tupa import constraints
from ucca import textutil

sys.path.insert(0, os.path.dirname(importlib.util.find_spec("smatch.smatch").origin))  # to find amr.py from smatch
from smatch import smatch
sys.path.pop(0)

prev_dir = os.getcwd()
os.chdir(os.path.dirname(importlib.util.find_spec("src.amr").origin))  # to find amr.peg
try:
    from src import amr as amr_lib
finally:
    os.chdir(prev_dir)

LABEL_ATTRIB = "label"
INSTANCE_OF = "instance-of"
PLACEHOLDER = re.compile("<[^>]*>")
VARIABLE_LABEL = "v"
UNKNOWN_LABEL = "Concept(amr-unknown)"
TERMINAL_TAGS = {constraints.EdgeTags.Terminal, constraints.EdgeTags.Punctuation}
ROLESET_PATTERN = re.compile("Concept\((.*)-(\d+)\)")
ROLES = {"Concept(ablate-01)": ("0", "1", "2", "3")}  # cache and fix for roles missing in PropBank


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
        super(Constraints, self).__init__(args, root_label=VARIABLE_LABEL, require_connected=True,
                                          require_implicit_childless=False, allow_root_terminal_children=True,
                                          possible_multiple_incoming=(), unique_outgoing={INSTANCE_OF},
                                          childless_incoming_trigger=INSTANCE_OF, unique_incoming=(),
                                          mutually_exclusive_outgoing=(), top_level=None)
        self.tag_rules.append(
            constraints.TagRule(trigger={constraints.Direction.incoming: "name"},
                                allowed={constraints.Direction.outgoing: re.compile(
                                    "^(%s|%s|op\d+)$" % (INSTANCE_OF, "|".join(TERMINAL_TAGS)))}))

    def allow_action(self, action, history):
        return True

    def _allow_edge(self, edge):
        return edge not in edge.parent.outgoing  # Prevent multiple identical edges between the same pair of nodes

    def allow_parent(self, node, tag):
        return not (node.implicit and tag in TERMINAL_TAGS or
                    not is_variable(node.label) and tag not in TERMINAL_TAGS) and \
               is_valid_arg(node, node.label, tag)

    def allow_child(self, node, tag):
        return is_concept(node.label) == (tag == INSTANCE_OF) and \
               (node.label == "Const(-)" or tag != "polarity") and \
               is_valid_arg(node, node.label, tag, is_parent=False)

    def allow_label(self, node, label):
        return (is_variable(label) or node.outgoing_tags <= TERMINAL_TAGS) and (
            not is_concept(label) or node.incoming_tags <= {INSTANCE_OF}) and (
            (label == "Const(-)") == (node.incoming_tags == {"polarity"})) and (
            not node.parents or
            is_valid_arg(node, label, *node.parents[0].outgoing_tags) and
            is_valid_arg(node, label, *node.parents[0].incoming_tags, is_parent=False)) and (
            TERMINAL_TAGS & node.outgoing_tags or is_variable(label) or
            not PLACEHOLDER.search(label))  # Prevent text placeholder in implicit node

    def allow_reduce(self, node):
        return node.text is not None or not is_variable(node.label) or INSTANCE_OF in node.outgoing_tags


def is_variable(label):
    return label == VARIABLE_LABEL


def is_concept(label):
    return label is not None and label.startswith("Concept(")


def is_valid_arg(node, label, *tags, is_parent=True):
    args = [t for t in tags if t.startswith("ARG") and (t.endswith("-of") != is_parent)]
    if not args:
        return True
    if label == VARIABLE_LABEL:
        for edge in node.outgoing:
            if edge.tag == INSTANCE_OF:
                node = edge.child
                label = node.label
                break
        else:
            return True
    label = resolve_label(node, label)
    valid_args = ROLES.get(label)
    if valid_args is None:
        try:
            roleset = pb.roleset(".".join(ROLESET_PATTERN.match(label).groups()))
            valid_args = tuple(r.attrib["n"] for r in roleset.findall("roles/role"))
        except (AttributeError, ValueError, TypeError):
            valid_args = ()
        ROLES[label] = valid_args
    return not valid_args or all(t.replace("-of", "").endswith(valid_args) for t in args)


def resolve_label(node, label=None, reverse=False):
    def _replace(old, new):  # replace only inside the label value/name
        new = new.strip('"()')
        if reverse:
            old, new = new, old
        replaceable = old and (len(old) > 2 or len(label) < 5)
        return re.sub(re.escape(old) + "(?![^<]*>|[^(]*\(|\d+$)", new, label) if replaceable else label

    def _related_forms(w):  # list of all derivationally related forms and their part of speech
        num_related = 0
        related = {None: w}
        while len(related) > num_related:
            num_related = len(related)
            related.update({v.synset().pos(): v.name() for x in related.values()
                            for l in wn.lemmas(x) for v in l.derivationally_related_forms()})
        return [(v, k) for k, v in related.items() if v != w]

    if label is None:
        try:
            label = node.label
        except AttributeError:
            label = node.attrib[LABEL_ATTRIB]
    if label != VARIABLE_LABEL:
        terminals = [c for c in node.children if getattr(c, "text", None)]
        if len(terminals) > 1:
            label = _replace("<t>", "".join(t.text for t in terminals))
        for i, terminal in enumerate(terminals):
            label = _replace("<t%d>" % i, terminal.text)
            label = _replace("<T%d>" % i, terminal.text.title())
            try:
                lemma = terminal.lemma
            except AttributeError:
                lemma = terminal.extra.get(textutil.LEMMA_KEY)
            if lemma == "-PRON-":
                lemma = terminal.text.lower()
            label = _replace("<l%d>" % i, lemma)
            label = _replace("<L%d>" % i, lemma.title())
            for form, pos in _related_forms(lemma):
                label = _replace("<%s%d>" % (pos, i), form)
    return label


# REPLACEMENTS = {"~": "about"}
