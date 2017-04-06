import importlib.util  # needed for amr.peg
import os
import re

from nltk.corpus import wordnet as wn, propbank as pb

from tupa import constraints
from ucca import textutil

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
COMMENT_PREFIX = "#"
ID_PATTERN = "#\s*::id\s+(\S+)"
TOK_PATTERN = "#\s*::(?:tok|snt)\s+(.*)"
DEP_PREFIX = ":"
TOP_DEP = ":top"
DEP_REPLACEMENT = {INSTANCE_OF: "instance"}
ALIGNMENT_PREFIX = "e."
ALIGNMENT_SEP = ","
LAYERS = {"wiki": ("wiki",),
          "numbers": ()}
ROLESET_PATTERN = re.compile("Concept\((.*)-(\d+)\)")
ROLES = {"Concept(ablate-01)": ("0", "1", "2", "3")}  # cache and fix for roles missing in PropBank


def parse(*args, **kwargs):
    return amr_lib.AMR(*args, **kwargs)


def is_variable(label):
    return label == VARIABLE_LABEL


def is_concept(label):
    return label is not None and label.startswith("Concept(")


def is_valid_arg(node, label, *tags, is_parent=True):
    if label is None:
        return True
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
    if label is None:
        return True
    label = resolve_label(node, label)
    valid_args = ROLES.get(label)
    if valid_args is None:
        try:
            roleset = pb.roleset(".".join(ROLESET_PATTERN.match(label).groups()))
            valid_args = tuple(r.attrib["n"] for r in roleset.findall("roles/role"))
        except (AttributeError, ValueError):
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
    if label not in (VARIABLE_LABEL, None):
        terminals = sorted([c for c in node.children if getattr(c, "text", None)],
                           key=lambda c: getattr(c, "index", getattr(c, "position", None)))
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
