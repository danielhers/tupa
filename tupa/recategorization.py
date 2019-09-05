import re
import string

from word2number import w2n

from tupa.constraints.util import OP
from .constraints.util import MONTHS, NUM_PATTERN, TOKEN_PLACEHOLDER, TOKEN_TITLE_PLACEHOLDER, LEMMA_PLACEHOLDER, \
    UNRESOLVED


def resolve(node, value, introduce_placeholders=False, conservative=False, is_node_label=True):
    """
    Replace any placeholder in node label/property with the corresponding terminals' text
    :param node: node whose label or property value is to be resolved
    :param value: the label or property value to resolve
    :param introduce_placeholders: if True, *introduce* placeholders into the label rather than resolving
    :param conservative: avoid replacement when risky due to multiple terminal children that could match
    :param is_node_label: is this a node label (not property value)
    :return: the resolved label, with or without placeholders (depending on the value of reverse)
    """
    if value is None:
        return None
    value = str(value)
    if value in UNRESOLVED:
        return value

    def _replace(old, new):  # replace only inside the label value/name
        new = new.strip('"()')
        if introduce_placeholders:
            old, new = new, old
        if old and (len(old) > 2 or len(value) < 5):
            try:
                return re.sub(re.escape(old) + r"(?![^<]*>|[^(]*\(|\d+$)", new, value, 1)
            except re.error:
                pass
        return value

    terminals = node.terminals
    if terminals:
        if not introduce_placeholders and NUM_PATTERN.match(value):  # numeric
            number = terminals_to_number(terminals)  # try replacing spelled-out numbers/months with digits
            if number is not None:
                value = str(number)
        else:
            if len(terminals) > 1:
                if introduce_placeholders or value.count(TOKEN_PLACEHOLDER) == 1:
                    value = _replace(TOKEN_PLACEHOLDER, "".join(map(lemmatize, terminals)))
                if introduce_placeholders or value.count(TOKEN_TITLE_PLACEHOLDER) == 1:
                    value = _replace(TOKEN_TITLE_PLACEHOLDER, "_".join(merge_punct(map(lemmatize, terminals))))
                if conservative:
                    terminals = ()
            for terminal in terminals:
                value = _replace(LEMMA_PLACEHOLDER, lemmatize(terminal))
                value = _replace(TOKEN_PLACEHOLDER, terminal.text)
    return value


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
    lemma = terminal.properties["lemma"]
    if lemma == "-PRON-":
        lemma = terminal.text
    return lemma.lower()


def merge_punct(tokens):
    """
    If a token starts/ends with punctuation, merge it with the previous/next token
    """
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


def compress_name(properties):
    """ Collapse :name (... / name) :op "..." into one string node """
    return {OP: "_".join(v for k, v in sorted(properties.items()))}


def expand_name(properties):
    """ Expand back names that have been collapsed """
    properties = dict(properties)
    op = properties.pop(OP, None)
    if op is not None:
        for i, op_i in enumerate(op.split("_"), start=1):
            properties[OP + str(i)] = op_i
    return properties
