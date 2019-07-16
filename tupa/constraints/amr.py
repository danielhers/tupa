import csv
import os
import re
import string
from collections import defaultdict
from operator import attrgetter

from word2number import w2n

from .validation import Constraints, Valid

PLACEHOLDER_PATTERN = re.compile(r"<[^>]*>")
NUM_PATTERN = re.compile(r"[+-]?\d+(\.\d+)?")
INT_PATTERN = re.compile(r"[+-]?(\d+)")
TOKEN_PLACEHOLDER = "<t>"
TOKEN_TITLE_PLACEHOLDER = "<T>"
LEMMA_PLACEHOLDER = "<l>"
NEGATION_PLACEHOLDER = "<n>"
LABEL_SEPARATOR = "|"  # after the separator there is the label category
PUNCTUATION_REMOVER = str.maketrans("", "", string.punctuation)
PREFIXED_RELATION_ENUM = ("op", "snt")
PREFIXED_RELATION_PREP = "prep"
PREFIXED_RELATION_PATTERN = re.compile(r"(?:(op|snt)\d+|(prep)-\w+)(-of)?")
PREFIXED_RELATION_SUBSTITUTION = r"\1\2\3"

# Specific edge labels (relations)
POLARITY = "polarity"
NAME = "name"
OP = "op"
MODE = "mode"
ARG2 = "ARG2"
VALUE = "value"
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
MINUS = "-"
MODES = ("expressive", "imperative", "interrogative")
DATE_ENTITY = "date-entity"
MONTHS = ("january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november",
          "december")
WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
SEASONS = ("winter", "fall", "spring", "summer")

# things to exclude from the graph because they are a separate task
EXTENSIONS = {
    "numbers": (),
    "urls": ("url-entity",),
}

NEGATIONS = {}
VERBALIZATION = defaultdict(dict)
ROLESETS = {}
CATEGORIES = {}


def read_resources():
    prev_dir = os.getcwd()
    if read_resources.done:
        return
    try:
        os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), "resources"))
        with open("negations.txt", encoding="utf-8") as f:
            NEGATIONS.update(csv.reader(f, delimiter=" "))
        with open("rolesets.txt", encoding="utf-8") as f:
            ROLESETS.update((l[0], tuple(l[1:])) for l in csv.reader(f))
        lines = []
        with open("wordnet.txt", encoding="utf-8") as f:
            lines += [re.findall(r'(\S):(\S+)', l) for l in f if l]
        with open("morph-verbalization-v1.01.txt", encoding="utf-8") as f:
            lines += [re.findall(r'::DERIV\S*-(\S)\S+ "(\S+)"', l) for l in f if l and l[0] != "#"]
        for pairs in lines:
            for prefix, word in pairs:
                VERBALIZATION[word].update(pairs)
        with open("verbalization-list-v1.06.txt", encoding="utf-8") as f:
            lines = (re.findall(r"(\S+) TO *(\S+ :\S+)? (\S+-\d+) *(\S+)?", l)[0] for l in f if l and l[0] not in "#D")
            for word, category, verb, suffix in lines:
                VERBALIZATION[word]["V"] = verb
                if category or suffix:
                    CATEGORIES[word] = category.replace(" ", "") + suffix
        with open("have-org-role-91-roles-v1.06.txt", encoding="utf-8") as f:
            # noinspection PyTypeChecker
            CATEGORIES.update(l.split()[::-1] for l in f if l and l[0] not in "#")
        with open("have-rel-role-91-roles-v1.06.txt", encoding="utf-8") as f:
            CATEGORIES.update(re.findall(r"(\S+) (\S+(?: [^:#]\S)*)", l)[0][::-1] for l in f if l and l[0] not in "#")
    finally:
        os.chdir(prev_dir)
    read_resources.done = True


read_resources.done = False


def is_int_in_range(label, s=None, e=None):
    m = INT_PATTERN.match(label)
    if not m:
        return Valid(False, "%s is not numeric" % label)
    num = int(m.group(1))
    return Valid(s is None or num >= s, "%s < %s" % (num, s)) and Valid(e is None or num <= e, "%s > %s" % (num, e))


def is_valid_arg(node, label, *labs, is_parent=True, is_concept=True):
    read_resources()
    if label is None:  # Not labeled yet or unlabeled parsing
        return True
    label = resolve_label(node, label, conservative=True, is_concept=is_concept)
    concept = label if is_concept else None
    const = label[1:-1] if label[0] == label[-1] == '"' else None
    if PLACEHOLDER_PATTERN.search(label):
        return True
    valid = Valid(message="%s incompatible as %s of %s" % (label, "parent" if is_parent else "child", ", ".join(labs)))
    if is_parent:  # node is a parent of the edge
        if {DAY, MONTH, YEAR, YEAR2, DECADE, WEEKDAY, QUARTER, CENTURY, SEASON, TIMEZONE}.intersection(labs):
            return valid(concept == DATE_ENTITY)
    elif const == MINUS:  # :polarity excl,b_isconst,b_const=-
        return valid({POLARITY, ARG2, VALUE}.issuperset(labs))
    elif POLARITY in labs:
        return valid(const == MINUS)
    elif MODE in labs:  # :mode excl,b_isconst,b_const=[interrogative|expressive|imperative]
        return valid(const in MODES)
    elif const in MODES:
        return valid(MODE in labs)
    elif DAY in labs:  # :day  a=date-entity,b_isconst,b_const=[...]
        return is_int_in_range(label, 1, 31)
    elif MONTH in labs:  # :month  a=date-entity,b_isconst,b_const=[1|2|3|4|5|6|7|8|9|10|11|12]
        return is_int_in_range(label, 1, 12)
    elif QUARTER in labs:  # :quarter  a=date-entity,b_isconst,b_const=[1|2|3|4]
        return is_int_in_range(label, 1, 4)
    elif {YEAR, YEAR2, DECADE, CENTURY}.intersection(labs):  # :year a=date-entity,b_isconst,b_const=[0-9]+
        return is_int_in_range(label)
    elif WEEKDAY in labs:  # :weekday  excl,a=date-entity,b=[monday|tuesday|wednesday|thursday|friday|saturday|sunday]
        return valid(concept in WEEKDAYS)
    elif concept in WEEKDAYS:
        return valid(WEEKDAY in labs)
    elif SEASON in labs:  # :season excl,a=date-entity,b=[winter|fall|spring|summer]+
        return valid(concept in SEASONS)

    if not concept or "-" not in concept:
        return True  # What follows is a check for predicate arguments, only relevant for predicates
    args = [t for t in labs if t.startswith("ARG") and (t.endswith("-of") != is_parent)]
    if not args:
        return True
    valid_args = ROLESETS.get(concept, ())
    return not valid_args or valid(all(t.replace("-of", "").endswith(valid_args) for t in args),
                                   "valid args: " + ", ".join(valid_args))


def resolve_label(node, label=None, reverse=False, conservative=False, is_concept=True):
    """
    Replace any placeholder in the node's label with the corresponding terminals' text, and remove label category suffix
    :param node: node whose label is to be resolved
    :param label: the label if not taken from the node directly
    :param reverse: if True, *introduce* placeholders and categories into the label rather than removing them
    :param conservative: avoid replacement when risky due to multiple terminal children that could match
    :param is_concept: is this a node label (not property value)
    :return: the resolved label, with or without placeholders and categories (depending on the value of reverse)
    """
    def _replace(old, new):  # replace only inside the label value/name
        new = new.strip('"()')
        if reverse:
            old, new = new, old
        replaceable = old and (len(old) > 2 or len(label) < 5)
        return re.sub(re.escape(old) + r"(?![^<]*>|[^(]*\(|\d+$)", new, label, 1) if replaceable else label

    read_resources()

    if label is None:
        label = node.label
    if label is not None:
        category = None
        if reverse:
            category = CATEGORIES.get(label)  # category suffix to append to label
        elif LABEL_SEPARATOR in label:
            label = label[:label.find(LABEL_SEPARATOR)]  # remove category suffix
        terminals = sorted([c for c in node.children if getattr(c, "text", None)], key=attrgetter("index"))
        if terminals:
            if not reverse and NUM_PATTERN.match(label):  # numeric label (always 1 unless "numbers" layer is on)
                number = terminals_to_number(terminals)  # try replacing spelled-out numbers/months with digits
                if number is not None:
                    label = str(number)
            else:
                if len(terminals) > 1:
                    if reverse or label.count(TOKEN_PLACEHOLDER) == 1:
                        label = _replace(TOKEN_PLACEHOLDER, "".join(t.text for t in terminals))
                    if reverse or label.count(TOKEN_TITLE_PLACEHOLDER) == 1:
                        label = _replace(TOKEN_TITLE_PLACEHOLDER, "_".join(merge_punct(t.text for t in terminals)))
                    if conservative:
                        terminals = ()
                for terminal in terminals:
                    lemma = lemmatize(terminal)
                    if lemma:
                        if reverse and category is None:
                            category = CATEGORIES.get(lemma)
                        label = _replace(LEMMA_PLACEHOLDER, lemma)
                    label = _replace(TOKEN_PLACEHOLDER, terminal.text)
                    label = _replace(TOKEN_TITLE_PLACEHOLDER, terminal.text.title())
                    negation = NEGATIONS.get(terminal.text)
                    if negation is not None:
                        label = _replace(NEGATION_PLACEHOLDER, negation)
                    if is_concept:
                        morph = VERBALIZATION.get(lemma)
                        if morph:
                            for prefix, value in morph.items():  # V: verb, N: noun, A: noun actor
                                label = _replace("<%s>" % prefix, value)
        if reverse and category:
            label += LABEL_SEPARATOR + category
    return label


def terminals_to_number(terminals):
    text = " ".join(t.text for t in terminals)
    try:  # first make sure it's not a number already
        float(text)
        return None
    except ValueError:
        pass
    # noinspection PyBroadException
    try:
        return w2n.word_to_num(text)
    except Exception:
        pass
    if len(terminals) == 1:
        try:
            return MONTHS.index(terminals[0].text.lower()) + 1
        except ValueError:
            pass


def lemmatize(terminal):
    lemma = terminal.get("lemma")
    if lemma == "-PRON-":
        lemma = terminal.text
    return lemma.translate(PUNCTUATION_REMOVER).lower() if lemma else None


# If a token starts/ends with punctuation, merge it with the previous/next token
def merge_punct(tokens):
    ret = list(tokens)
    while len(ret) > 1:
        for i, token in enumerate(ret):
            s, e = i, i + 1
            if len(token):
                if e < len(ret) and token.endswith(tuple(string.punctuation)):
                    e += 1
                if s and token.startswith(tuple(string.punctuation)):
                    s -= 1
            if s + 1 < e:
                ret[s:e] = ["".join(ret[s:e])]
                break
        else:
            break
    return ret


class AmrConstraints(Constraints):
    def __init__(self, **kwargs):
        super().__init__(multigraph=True, require_implicit_childless=False, allow_orphan_terminals=True,
                         childless_incoming_trigger={POLARITY, CENTURY, DECADE, "polite", "li"}, **kwargs)

    def allow_action(self, action, history):
        return True

    def allow_edge(self, edge):  # Prevent multiple identical edges between the same pair of nodes
        return edge.lab in PREFIXED_RELATION_ENUM or edge not in edge.parent.outgoing

    def allow_parent(self, node, lab):
        return not lab or is_valid_arg(node, node.label, lab)

    def allow_child(self, node, lab):
        return not lab or is_valid_arg(node, node.label, lab, is_parent=False)

    def allow_label(self, node, label):
        return not node.parents or \
               is_valid_arg(node, label, *node.outgoing_labs) and \
               is_valid_arg(node, label, *node.incoming_labs, is_parent=False)
