import csv
import importlib.util  # needed for amr.peg
import os
import re
import string

from nltk.corpus import propbank as pb
from ucca import textutil
from word2number import w2n

from tupa import constraints

prev_dir = os.getcwd()
try:
    os.chdir(os.path.dirname(importlib.util.find_spec("src.amr").origin))  # to find amr.peg
    from src import amr as amr_lib
finally:
    os.chdir(prev_dir)

NEGATIONS = {}
MORPH_VERBALIZATION = {}

try:
    os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), "resources"))
    with open("negations.txt") as f:
        NEGATIONS.update(csv.reader(f, delimiter=" "))
    with open("morph-verbalization-v1.01.txt") as f:
        # noinspection PyTypeChecker
        lines = (re.findall(r'::DERIV\S*-(\S)\S+ "(\S+)"', l) for l in f if l and l[0] != "#")
        MORPH_VERBALIZATION.update({w: l for l in lines for (_, w) in l})
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
SKIP_TOKEN = re.compile("[<>@]+")
LABEL_ATTRIB = "label"
KNOWN_LABELS = set()  # used to avoid escaping when unnecessary
PUNCTUATION_REMOVER = str.maketrans("", "", string.punctuation)
PREFIXED_RELATION_ENUM = ("op", "snt")
PREFIXED_RELATION_PREP = "prep"
PREFIXED_RELATION_PATTERN = re.compile("(?:(op|snt)\d+|(prep)-\w+)(-of)?")
PREFIXED_RELATION_SUBSTITUTION = r"\1\2\3"

# Specific edge labels (relations)
INSTANCE = "instance"
POLARITY = "polarity"
MODE = "mode"
ARG2 = "ARG2"
VALUE = "value"
WIKI = "wiki"
DAY = "day"
MONTH = "month"
YEAR = "year"
YEAR2 = "year2"
DECADE = "decade"
WEEKDAY = "weekday"
QUARTER = "quarter"
CENTURY = "century"
SEASON = "season"
TIMEZONE = "timezone"

# Specific node labels
CONST = "Const"
CONST_MINUS = CONST + "(-)"
CONST_MODES = (CONST + "(expressive)", CONST + "(imperative)", CONST + "(interrogative)")
CONCEPT = "Concept"
UNKNOWN_LABEL = CONCEPT + "(amr-unknown)"
DATE_ENTITY = CONCEPT + "(date-entity)"
ROLESET_PATTERN = re.compile(CONCEPT + "\((.*)-(\d+)\)")
ROLES = {  # cache + fix for roles missing in PropBank
    CONCEPT + "(ablate-01)": ("0", "1", "2", "3"),
    CONCEPT + "(play-11)": ("0", "1", "2", "3"),
    CONCEPT + "(raise-02)": ("0", "1", "2", "3"),
}
NUM = "Num"
MONTHS = ("january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december")
WEEKDAYS = (CONCEPT + "(monday)", CONCEPT + "(tuesday)", CONCEPT + "(wednesday)",
            CONCEPT + "(thursday)", CONCEPT + "(friday)", CONCEPT + "(saturday)", CONCEPT + "(sunday)")
SEASONS = (CONCEPT + "(winter)", CONCEPT + "(fall)", CONCEPT + "(spring)", CONCEPT + "(summer)")

# things to exclude from the graph because they are a separate task
LAYERS = {
    WIKI: (":" + WIKI,),
    "numbers": (),
    "urls": (amr_lib.Concept("url-entity"),),
}


def parse(*args, **kwargs):
    return amr_lib.AMR(*args, **kwargs)


def is_concept(label):
    return label is not None and label.startswith(CONCEPT + "(")


def is_int_in_range(label, s=None, e=None):
    m = re.match(NUM + "\(-?(\d+)\)", label)
    if not m:
        return False
    num = int(m.group(1))
    return (s is None or num >= s) and (e is None or num <= e)


def is_valid_arg(node, label, *tags, is_parent=True):
    if label is None:
        return True
    label = resolve_label(node, label)
    if PLACEHOLDER.search(label):
        return True
    if is_parent:  # node is a parent of the edge
        if {DAY, MONTH, YEAR, YEAR2, DECADE, WEEKDAY, QUARTER, CENTURY, SEASON, TIMEZONE}.intersection(tags):
            return label == DATE_ENTITY
    elif label == CONST_MINUS:  # :polarity excl,b_isconst,b_const=-
        return {POLARITY, ARG2, VALUE}.issuperset(tags)
    elif POLARITY in tags:
        return label == CONST_MINUS
    elif MODE in tags:  # :mode excl,b_isconst,b_const=[interrogative|expressive|imperative]
        return label in CONST_MODES
    elif label in CONST_MODES:
        return MODE in tags
    elif WIKI in tags:  # :wiki b_isconst (:value and :timezone are not really always const)
        return label.startswith(CONST + "(")
    elif DAY in tags:  # :day  a=date-entity,b_isconst,b_const=[...]
        return is_int_in_range(label, 1, 31)
    elif MONTH in tags:  # :month  a=date-entity,b_isconst,b_const=[1|2|3|4|5|6|7|8|9|10|11|12]
        return is_int_in_range(label, 1, 12)
    elif QUARTER in tags:  # :quarter  a=date-entity,b_isconst,b_const=[1|2|3|4]
        return is_int_in_range(label, 1, 4)
    elif {YEAR, YEAR2, DECADE, CENTURY}.intersection(tags):  # :year a=date-entity,b_isconst,b_const=[0-9]+
        return is_int_in_range(label)
    elif WEEKDAY in tags:  # :weekday  excl,a=date-entity,b=[monday|tuesday|wednesday|thursday|friday|saturday|sunday]
        return label in WEEKDAYS
    elif label in WEEKDAYS:
        return WEEKDAY in tags
    elif SEASON in tags:  # :season excl,a=date-entity,b=[winter|fall|spring|summer]+
        return label in SEASONS

    args = [t for t in tags if t.startswith("ARG") and (t.endswith("-of") != is_parent)]
    if not args:
        return True
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
                    morph = MORPH_VERBALIZATION.get(terminal.text)
                    if morph is not None:
                        for prefix, value in morph:  # V: verb, N: noun, A: noun actor
                            label = _replace("<%s%d>" % (prefix, i), value)
    if reverse:
        KNOWN_LABELS.add(label)
    return label


def terminals_to_number(terminals):
    # noinspection PyBroadException
    text = " ".join(t.text for t in terminals)
    try:  # first make sure it's not a number already
        float(text)
        return None
    except ValueError:
        pass
    # noinspection PyBroadException
    try:
        return w2n.word_to_num(text)
    except:
        pass
    if len(terminals) == 1:
        try:
            return MONTHS.index(terminals[0].text.lower()) + 1
        except ValueError:
            pass

# REPLACEMENTS = {"~": "about"}
