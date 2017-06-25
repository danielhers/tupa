import csv
import importlib.util  # needed for amr.peg
import os
import re
import string

from nltk.corpus import propbank as pb
from ucca import textutil
from word2number import w2n

from tupa import constraints

NEGATIONS = {}

prev_dir = os.getcwd()
try:
    os.chdir(os.path.dirname(importlib.util.find_spec("src.amr").origin))  # to find amr.peg
    from src import amr as amr_lib
finally:
    os.chdir(prev_dir)

try:
    os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), "resources"))
    with open("negations.txt") as f:
        NEGATIONS.update(csv.reader(f, delimiter=" "))
finally:
    os.chdir(prev_dir)

LAYERS = {  # things to exclude from the graph because they are a separate task
    "wiki": (":wiki",),
    "numbers": (),
    "urls": (amr_lib.Concept("url-entity"),),
}
TERMINAL_TAGS = {constraints.EdgeTags.Terminal, constraints.EdgeTags.Punctuation}
COMMENT_PREFIX = "#"
ID_PATTERN = "#\s*::id\s+(\S+)"
TOK_PATTERN = "#\s*::(?:tok|snt)\s+(.*)"
DEP_PREFIX = ":"
TOP_DEP = ":top"
ALIGNMENT_PREFIX = "e."
ALIGNMENT_SEP = ","
PLACEHOLDER = re.compile("<[^>]*>")
SKIP_TOKEN = re.compile("[<>@]+")
LABEL_ATTRIB = "label"
INSTANCE = "instance"
CONCEPT = "Concept"
NUM = "Num"
UNKNOWN_LABEL = CONCEPT + "(amr-unknown)"
ROLESET_PATTERN = re.compile(CONCEPT + "\((.*)-(\d+)\)")
ROLES = {  # cache + fix for roles missing in PropBank
    CONCEPT + "(ablate-01)": ("0", "1", "2", "3"),
    CONCEPT + "(play-11)": ("0", "1", "2", "3"),
    CONCEPT + "(raise-02)": ("0", "1", "2", "3"),
}
MONTHS = ("january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december")
KNOWN_LABELS = set()  # used to avoid escaping when unnecessary
PUNCTUATION_REMOVER = str.maketrans("", "", string.punctuation)
PREFIXED_RELATION_ENUM = ("op", "snt")
PREFIXED_RELATION_PREP = "prep"
PREFIXED_RELATION_PATTERN = re.compile("(?:(op|snt)\d+|(prep)-\w+)(-of)?")
PREFIXED_RELATION_SUBSTITUTION = r"\1\2\3"


def parse(*args, **kwargs):
    return amr_lib.AMR(*args, **kwargs)


def is_concept(label):
    return label is not None and label.startswith(CONCEPT + "(")


def is_valid_arg(node, label, *tags, is_parent=True):
    if label is None:
        return True
    if not is_parent and node.label == "Const(-)":
        return {"polarity", "ARG2", "value"}.issuperset(tags)
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
    """
    Replace any placeholder in the node's label with the corresponding terminals' text
    :param node: node whose label is to be resolved
    :param label: the label if not taken from the node directly
    :param reverse: if True, *introduce* placeholders into the label rather than removing them
    :return: the resolved label, with or without placeholders (depending on the value of reverse)
    """
    def _replace(old, new):  # replace only inside the label value/name
        new = new.strip('"()')
        if reverse:
            old, new = new, old
        replaceable = old and (len(old) > 2 or len(label) < 5)
        return re.sub(re.escape(old) + "(?![^<]*>|[^(]*\(|\d+$)", new, label) if replaceable else label

    if label is None:
        try:
            label = node.label
        except AttributeError:
            label = node.attrib.get(LABEL_ATTRIB)
    if label is not None and not (reverse and label in KNOWN_LABELS):
        children = [c.children[0] if c.tag == "PNCT" else c for c in node.children]
        terminals = sorted([c for c in children if getattr(c, "text", None)],
                           key=lambda c: getattr(c, "index", getattr(c, "position", None)))
        if terminals:
            if not reverse and label.startswith(NUM + "("):  # numeric label (always 1 unless "numbers" layer is on)
                number = terminals_to_number(terminals)  # try replacing spelled-out numbers/months with digits
                if number is not None:
                    label = "%s(%s)" % (NUM, number)
            else:
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
                    lemma = lemma.translate(PUNCTUATION_REMOVER)
                    label = _replace("<l%d>" % i, lemma)
                    label = _replace("<L%d>" % i, lemma.title())
                    negation = NEGATIONS.get(terminal.text)
                    if negation is not None:
                        label = _replace("<n%d>" % i, negation)
    return label


def terminals_to_number(terminals):
    # noinspection PyBroadException
    try:
        return w2n.word_to_num(" ".join(t.text for t in terminals))
    except:
        pass
    if len(terminals) == 1:
        try:
            return MONTHS.index(terminals[0].text.lower()) + 1
        except ValueError:
            pass

# REPLACEMENTS = {"~": "about"}
