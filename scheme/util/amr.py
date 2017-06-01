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


TERMINAL_TAGS = {constraints.EdgeTags.Terminal, constraints.EdgeTags.Punctuation}
COMMENT_PREFIX = "#"
ID_PATTERN = "#\s*::id\s+(\S+)"
TOK_PATTERN = "#\s*::(?:tok|snt)\s+(.*)"
DEP_PREFIX = ":"
TOP_DEP = ":top"
ALIGNMENT_PREFIX = "e."
ALIGNMENT_SEP = ","
PLACEHOLDER = re.compile("<[^>]*>")
LABEL_ATTRIB = "label"
INSTANCE = "instance"
CONCEPT = "Concept"
UNKNOWN_LABEL = CONCEPT + "(amr-unknown)"
LAYERS = {"wiki": (":wiki",),
          "numbers": (),
          "urls": (CONCEPT + "(url-entity)",)}
ROLESET_PATTERN = re.compile(CONCEPT + "\((.*)-(\d+)\)")
ROLES = {  # cache + fix for roles missing in PropBank
    CONCEPT + "(ablate-01)": ("0", "1", "2", "3"),
    CONCEPT + "(play-11)": ("0", "1", "2", "3"),
    CONCEPT + "(raise-02)": ("0", "1", "2", "3"),
}


def parse(*args, **kwargs):
    return amr_lib.AMR(*args, **kwargs)


def is_concept(label):
    return label is not None and label.startswith(CONCEPT + "(")


def is_valid_arg(node, label, *tags, is_parent=True):
    if label is None:
        return True
    if not is_parent and node.label == "Const(-)":
        return {"polarity", "ARG2"}.issuperset(tags)
    args = [t for t in tags if t.startswith("ARG") and (t.endswith("-of") != is_parent)]
    if not args:
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
            label = node.attrib.get(LABEL_ATTRIB)
    if label is not None:
        children = [c.children[0] if c.tag == "PNCT" else c for c in node.children]
        terminals = sorted([c for c in children if getattr(c, "text", None)],
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
