import csv
import importlib.util  # needed for amr.peg
import os
import re
import string
from collections import defaultdict

import spotlight
from requests.exceptions import ConnectionError
from spotlight import SpotlightException
from ucca import layer1
from ucca import textutil
from ucca.convert import to_text
from word2number import w2n

from ..constraints import Valid

prev_dir = os.getcwd()
try:
    os.chdir(os.path.dirname(importlib.util.find_spec("src.amr").origin))  # to find amr.peg
    from src import amr as amr_lib
finally:
    os.chdir(prev_dir)

TERMINAL_DEP = layer1.EdgeTags.Terminal
PUNCTUATION_DEP = layer1.EdgeTags.Punctuation
PUNCTUATION_LABEL = layer1.NodeTags.Punctuation
TERMINAL_TAGS = {TERMINAL_DEP, PUNCTUATION_DEP}
COMMENT_PREFIX = "#"
ID_PATTERN = re.compile("#\s*::id\s+(\S+)")
TOK_PATTERN = re.compile("#\s*::(?:tok|snt)\s+(.*)")
DEP_PREFIX = ":"
TOP_DEP = ":top"
ALIGNMENT_PREFIX = "e."
ALIGNMENT_SEP = ","
PLACEHOLDER_PATTERN = re.compile(r"<[^>]*>")
SKIP_TOKEN_PATTERN = re.compile(r"[<>@]+")
NUM_PATTERN = re.compile(r"[+-]?\d+(\.\d+)?")
TOKEN_PLACEHOLDER = "<t>"
TOKEN_TITLE_PLACEHOLDER = "<T>"
LEMMA_PLACEHOLDER = "<l>"
NEGATION_PLACEHOLDER = "<n>"
WIKIFICATION_PLACEHOLDER = "<w>"
LABEL_ATTRIB = "label"
LABEL_SEPARATOR = "|"  # after the separator there is the label category
PUNCTUATION_REMOVER = str.maketrans("", "", string.punctuation)
PREFIXED_RELATION_ENUM = ("op", "snt")
PREFIXED_RELATION_PREP = "prep"
PREFIXED_RELATION_PATTERN = re.compile("(?:(op|snt)\d+|(prep)-\w+)(-of)?")
PREFIXED_RELATION_SUBSTITUTION = r"\1\2\3"

# Specific edge labels (relations)
INSTANCE = "instance"
POLARITY = "polarity"
NAME = "name"
OP = "op"
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
CONCEPT = "Concept"
NUM = "Num"
MINUS = "-"
UNKNOWN_LABEL = CONCEPT + "(name)"
MODES = ("expressive", "imperative", "interrogative")
DATE_ENTITY = "date-entity"
MONTHS = ("january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november",
          "december")
WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
SEASONS = ("winter", "fall", "spring", "summer")

# things to exclude from the graph because they are a separate task
EXTENSIONS = {
    WIKI: (),
    "numbers": (),
    "urls": (amr_lib.Concept("url-entity"),),
}

NEGATIONS = {}
VERBALIZATION = defaultdict(dict)
ROLESETS = {}
CATEGORIES = {}


def read_resources():
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
            CATEGORIES.update(l.split()[::-1] for l in f if l and l[0] not in "#")
        with open("have-rel-role-91-roles-v1.06.txt", encoding="utf-8") as f:
            CATEGORIES.update(re.findall(r"(\S+) (\S+(?: [^:#]\S)*)", l)[0][::-1] for l in f if l and l[0] not in "#")
    finally:
        os.chdir(prev_dir)


def parse(*args, **kwargs):
    return amr_lib.AMR(*args, **kwargs)


def is_concept(label):
    return label is not None and label.startswith(CONCEPT)


def is_int_in_range(label, s=None, e=None):
    m = re.match(NUM + "\(-?(\d+)\)", label)
    if not m:
        return Valid(False, "%s is not numeric" % label)
    num = int(m.group(1))
    return Valid(s is None or num >= s, "%s < %s" % (num, s)) and Valid(e is None or num <= e, "%s > %s" % (num, e))


def is_valid_arg(node, label, *tags, is_parent=True):
    if label is None:
        return True
    label = resolve_label(node, label, conservative=True)
    concept = label[len(CONCEPT) + 1:-1] if label.startswith(CONCEPT) else None
    const = label[len(CONST) + 1:-1] if label.startswith(CONST) else None
    if PLACEHOLDER_PATTERN.search(label):
        return True
    valid = Valid(message="%s incompatible as %s of %s" % (label, "parent" if is_parent else "child", ", ".join(tags)))
    if is_parent:  # node is a parent of the edge
        if {DAY, MONTH, YEAR, YEAR2, DECADE, WEEKDAY, QUARTER, CENTURY, SEASON, TIMEZONE}.intersection(tags):
            return valid(concept == DATE_ENTITY)
    elif const == MINUS:  # :polarity excl,b_isconst,b_const=-
        return valid({POLARITY, ARG2, VALUE, WIKI}.issuperset(tags))
    elif POLARITY in tags:
        return valid(const == MINUS)
    elif MODE in tags:  # :mode excl,b_isconst,b_const=[interrogative|expressive|imperative]
        return valid(const in MODES)
    elif const in MODES:
        return valid(MODE in tags)
    elif WIKI in tags:  # :wiki b_isconst (:value and :timezone are not really always const)
        return valid(const == MINUS)
    elif DAY in tags:  # :day  a=date-entity,b_isconst,b_const=[...]
        return is_int_in_range(label, 1, 31)
    elif MONTH in tags:  # :month  a=date-entity,b_isconst,b_const=[1|2|3|4|5|6|7|8|9|10|11|12]
        return is_int_in_range(label, 1, 12)
    elif QUARTER in tags:  # :quarter  a=date-entity,b_isconst,b_const=[1|2|3|4]
        return is_int_in_range(label, 1, 4)
    elif {YEAR, YEAR2, DECADE, CENTURY}.intersection(tags):  # :year a=date-entity,b_isconst,b_const=[0-9]+
        return is_int_in_range(label)
    elif WEEKDAY in tags:  # :weekday  excl,a=date-entity,b=[monday|tuesday|wednesday|thursday|friday|saturday|sunday]
        return valid(concept in WEEKDAYS)
    elif concept in WEEKDAYS:
        return valid(WEEKDAY in tags)
    elif SEASON in tags:  # :season excl,a=date-entity,b=[winter|fall|spring|summer]+
        return valid(concept in SEASONS)
    elif PUNCTUATION_DEP in tags:
        return valid(label == PUNCTUATION_LABEL)
    elif label == PUNCTUATION_LABEL:
        return valid(tags == {PUNCTUATION_DEP})

    if not concept or "-" not in concept:
        return True  # What follows is a check for predicate arguments, only relevant for predicates
    args = [t for t in tags if t.startswith("ARG") and (t.endswith("-of") != is_parent)]
    if not args:
        return True
    valid_args = ROLESETS.get(concept, ())
    return not valid_args or valid(all(t.replace("-of", "").endswith(valid_args) for t in args),
                                   "valid args: " + ", ".join(valid_args))


def resolve_label(node, label=None, reverse=False, conservative=False):
    """
    Replace any placeholder in the node's label with the corresponding terminals' text, and remove label category suffix
    :param node: node whose label is to be resolved
    :param label: the label if not taken from the node directly
    :param reverse: if True, *introduce* placeholders and categories into the label rather than removing them
    :param conservative: avoid replacement when risky due to multiple terminal children that could match
    :return: the resolved label, with or without placeholders and categories (depending on the value of reverse)
    """
    def _replace(old, new):  # replace only inside the label value/name
        new = new.strip('"()')
        if reverse:
            old, new = new, old
        replaceable = old and (len(old) > 2 or len(label) < 5)
        return re.sub(re.escape(old) + "(?![^<]*>|[^(]*\(|\d+$)", new, label, 1) if replaceable else label

    if label is None:
        try:
            label = node.label
        except AttributeError:
            label = node.attrib.get(LABEL_ATTRIB)
    if label is not None:
        category = None
        if reverse:
            category = CATEGORIES.get(label)  # category suffix to append to label
        elif LABEL_SEPARATOR in label:
            label = label[:label.find(LABEL_SEPARATOR)]  # remove category suffix
        children = [c.children[0] if c.tag == "PNCT" else c for c in node.children]
        terminals = sorted([c for c in children if getattr(c, "text", None)],
                           key=lambda c: getattr(c, "index", getattr(c, "position", None)))
        if terminals:
            if not reverse and label.startswith(NUM):  # numeric label (always 1 unless "numbers" layer is on)
                number = terminals_to_number(terminals)  # try replacing spelled-out numbers/months with digits
                if number is not None:
                    label = NUM + "(%s)" % number
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
                    if reverse and category is None:
                        category = CATEGORIES.get(lemma)
                    label = _replace(LEMMA_PLACEHOLDER, lemma)
                    label = _replace(TOKEN_PLACEHOLDER, terminal.text)
                    label = _replace(TOKEN_TITLE_PLACEHOLDER, terminal.text.title())
                    negation = NEGATIONS.get(terminal.text)
                    if negation is not None:
                        label = _replace(NEGATION_PLACEHOLDER, negation)
                    if label.startswith(CONCEPT):
                        morph = VERBALIZATION.get(lemma)
                        if morph is not None:
                            for prefix, value in morph.items():  # V: verb, N: noun, A: noun actor
                                label = _replace("<%s>" % prefix, value)
                    elif label.startswith('"') and (reverse and not PLACEHOLDER_PATTERN.search(label) or
                                                    not reverse and WIKIFICATION_PLACEHOLDER in label):
                        try:
                            label = _replace(WIKIFICATION_PLACEHOLDER, WIKIFIER.wikify_terminal(terminal))
                        except (ValueError, IOError):
                            pass
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
    except:
        pass
    if len(terminals) == 1:
        try:
            return MONTHS.index(terminals[0].text.lower()) + 1
        except ValueError:
            pass


def lemmatize(terminal):
    try:
        lemma = terminal.lemma
    except AttributeError:
        lemma = terminal.extra.get(textutil.LEMMA_KEY)
    if lemma == "-PRON-":
        lemma = terminal.text.lower()
    return lemma.translate(PUNCTUATION_REMOVER)


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


class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        ret = self[key] = self.default_factory(key)
        return ret


class Wikifier:
    def __init__(self, enabled=True):
        self.address = os.environ.get("SPOTLIGHT_ADDRESS", "http://model.dbpedia-spotlight.org/en/annotate")
        self.confidence = float(os.environ.get("SPOTLIGHT_CONFIDENCE", 0.3))
        self.text = None
        self.spots = ()
        self.passage_texts = keydefaultdict(lambda passage: to_text(passage, sentences=False)[0])
        self.enabled = enabled

    def wikify_terminal(self, terminal):
        text = self.passage_texts[terminal.root]
        return self.wikify_text(text, text.find(terminal.text))

    def wikify_text(self, text, offset):
        if not self.enabled:
            raise ValueError("Wikifier is disabled")
        error = ValueError("Failed to wikify '%s' offset %d" % (text, offset))
        if self.text != text:
            self.text = text
            try:
                self.spots = spotlight.annotate(self.address, text, confidence=self.confidence) if text.strip() else ()
            except (ValueError, SpotlightException, ConnectionError) as e:
                raise error from e
        for spot in self.spots:
            if spot["offset"] == offset:
                return '"%s"' % spot["URI"].replace("http://dbpedia.org/resource/", "")
        raise error

    def wikify_node(self, text, node, name):
        try:
            node_text = " ".join(t.text for t in node.get_terminals()) or \
                        " ".join(filter(None, (n.attrib.get(LABEL_ATTRIB) for n in name.children))).replace('"', "")
            return self.wikify_text(text, text.find(node_text))
        except ValueError:
            return "-"

    def wikify_passage(self, passage):
        l1 = passage.layer(layer1.LAYER_ID)
        for node in l1.all:
            name = wiki = None
            for edge in node:
                if edge.tag == NAME:
                    name = edge
                elif edge.tag == WIKI:
                    wiki = edge
            if wiki is not None:
                node.remove(wiki)
                if name is not None:
                    l1.add_fnode(node, WIKI).attrib[LABEL_ATTRIB] = self.wikify_node(
                        self.passage_texts[passage], node, name.child)


read_resources()
WIKIFIER = Wikifier()
