import re
from ucca import layer0
from ucca.layer1 import EdgeTags
from ucca.textutil import Attr

from tupa.config import Config, FEATURE_PROPERTIES

FEATURE_ELEMENT_PATTERN = re.compile(r"([sba])(\d)([lrLR]*)([%s]*)" % FEATURE_PROPERTIES)
FEATURE_TEMPLATE_PATTERN = re.compile(r"^(%s)+$" % FEATURE_ELEMENT_PATTERN.pattern)
NON_NUMERIC = "wmtudencpAT#^$"


class FeatureTemplate:
    """
    A feature template in parsed form, ready to be used for value calculation
    """

    def __init__(self, name, elements):
        """
        :param name: name of the feature in the short-hand form, to be used for the dictionary
        :param elements: collection of FeatureElement objects that represent the actual feature
        """
        self.name = name
        self.elements = elements
        for prev, elem in zip(self.elements[:-1], self.elements[1:]):
            if not prev.properties:
                assert prev.previous is None, "More than one consecutive empty element: %s%s" % (prev.previous, prev)
                elem.previous = prev

    def __str__(self):
        return self.name

    def extract(self, state, default=None, indexed=(), as_tuples=False, node_dropout=0, hierarchical=False):
        try:
            return [value for element in self.elements for value in element.extract(state, default, indexed, as_tuples,
                                                                                    node_dropout=node_dropout,
                                                                                    hierarchical=hierarchical)]
        except ValueError:
            return None


class FeatureTemplateElement:
    """
    One element in the values of a feature, e.g. from one node
    """

    def __init__(self, source, index, relatives, properties, omit_features=None):
        """
        :param source: where to take the data from:
                           s: stack nodes
                           b: buffer nodes
                           a: past actions
        :param index: non-negative integer, the index of the element in the stack, buffer or list
                           of past actions (in the case of stack and actions, indexing from the end)
        :param relatives: string in [lrLR]*, to select a descendant/parent of the node instead:
                           l: leftmost child
                           r: rightmost child
                           L: leftmost parent
                           R: rightmost parent
        :param properties: the actual values to choose, if available (else omit feature), out of:
                           w: node text
                           m: node lemma
                           t: node fine POS tag
                           u: node coarse/universal POS tag
                           d: node dependency relation
                           h: node height
                           e: tag of first incoming edge / action tag
                           n: node label
                           c: node label category suffix
                           p: unique separator punctuation between nodes
                           q: count of any separator punctuation between nodes
                           x: numeric value of gap type
                           y: sum of gap lengths
                           A: action type label
                           P: number of parents
                           C: number of children
                           I: number of implicit children
                           E: number of remote children
                           M: number of remote parents
                           N: numeric value of named entity IOB
                           T: named entity type
                           #: word shape
                           ^: word prefix (one character)
                           $: word suffix (three characters)
                           If empty,
                             If the next node comes with the "x" property, the value will be 1 if there is an edge from
                             this node to the next one in the template, or 0 otherwise.
                             If the next node comes with the "e" property, the edge with this node will be considered.
                             If the next node comes with the "d" property, the value will be the dependency distance
                             between the head terminals of the nodes.
        """
        self.source = source
        self.index = int(index)
        self.relatives = relatives
        self.properties = properties.translate(str.maketrans("", "", omit_features)) if omit_features else properties
        self.node = self.previous = None
        self.getters = [prop_getter(prop, self.source) for prop in self.properties]

    def __str__(self):
        return self.str + self.properties

    @property
    def str(self):
        return (self.previous.str if self.previous else "") + self.source + str(self.index) + self.relatives

    def __eq__(self, other):
        return self.source == other.source and self.index == other.index and self.relatives == other.relatives

    def set_node(self, state, node_dropout=0):
        self.node = None
        if state is None or node_dropout and node_dropout > Config().random.random_sample():
            return
        try:
            if self.source == "s":
                self.node = state.stack[-1 - self.index]
            elif self.source == "b":
                self.node = state.buffer[self.index]
            else:  # source == "a"
                self.node = state.actions[-1 - self.index]
            for relative in self.relatives:
                nodes = self.node.parents if relative.isupper() else self.node.children
                if relative.lower() == "r":
                    if len(nodes) == 1:
                        raise ValueError("Avoiding identical right and left relatives")
                    self.node = nodes[-1]
                else:  # relative.lower() == "l"
                    self.node = nodes[0]
        except (IndexError, TypeError, AttributeError, IndexError, ValueError):
            if Config().args.missing_node_features or node_dropout and node_dropout > Config().random.random_sample():
                self.node = None

    def extract(self, state, default, indexed, as_tuples, node_dropout=0, hierarchical=False):
        self.set_node(state, node_dropout=node_dropout)
        for prop, getter in zip(self.properties, self.getters):
            if indexed and not self.is_numeric(prop):
                if prop == indexed[0]:
                    getter = NODE_PROP_GETTERS["j" if hierarchical else "i"]
                elif prop in indexed[1:]:
                    continue
            value = self.get_prop(state, prop, getter, default)
            yield (self, prop, value) if as_tuples else value

    def get_prop(self, state, prop, getter, default):
        value = calc(self.node, state, prop, getter, self.previous)
        if value is None:
            if default is None:
                raise ValueError("Value does not exist, and no default given")
            value = default
        return value

    def is_numeric(self, prop):
        return bool(prop not in NON_NUMERIC or (prop == "d" and self.previous))


class FeatureExtractor:
    """
    Object to extract features from the parser state to be used in action classification
    """

    def __init__(self, feature_templates=(), params=None, omit_features=None):
        assert all(FEATURE_TEMPLATE_PATTERN.match(f) for f in feature_templates), \
            "Features do not match pattern: " + ", ".join(
                f for f in feature_templates if not FEATURE_TEMPLATE_PATTERN.match(f))
        # convert the list of features textual descriptions to the actual fields
        self.feature_templates = [
            FeatureTemplate(feature_name, tuple(FeatureTemplateElement(*m.group(1, 2, 3, 4), omit_features)
                                                for m in re.finditer(FEATURE_ELEMENT_PATTERN, feature_name)))
            for feature_name in feature_templates]
        self.params = {} if params is None else params
        self.omit_features = omit_features

    def extract_features(self, state):
        """
        Calculate feature values according to current state
        :param state: current state of the parser
        """
        raise NotImplementedError()

    def init_features(self, state):
        """
        Calculate feature values for initial state
        :param state: initial state of the parser
        """
        pass

    def init_param(self, key):
        pass

    def finalize(self):
        return self

    def unfinalize(self):
        pass

    def save(self, filename, save_init=True):
        pass

    def load(self, filename, order=None):
        pass

    def all_features(self):
        return sorted(list(map(str, self.feature_templates)))


EDGE_PRIORITY = {tag: i for i, tag in enumerate((
    EdgeTags.Center,
    EdgeTags.Connector,
    EdgeTags.ParallelScene,
    EdgeTags.Process,
    EdgeTags.State,
    EdgeTags.Participant,
    EdgeTags.Adverbial,
    EdgeTags.Time,
    EdgeTags.Elaborator,
    EdgeTags.Relator,
    EdgeTags.Function,
    EdgeTags.Linker,
    EdgeTags.LinkRelation,
    EdgeTags.LinkArgument,
    EdgeTags.Ground,
    EdgeTags.Terminal,
    EdgeTags.Punctuation,
))}


def head_terminal(node, *_):
    return head_terminal_height(node)


def height(node, *_):
    return head_terminal_height(node, return_height=True)


def static_vars(**kwargs):
    def decorate(func):
        for k, v in kwargs.items():
            setattr(func, k, v)
        return func
    return decorate


MAX_HEIGHT = 30


@static_vars(node=None, head_terminal=None, height=None)
def head_terminal_height(node, return_height=False):
    if node is not head_terminal_height.node:
        head_terminal_height.node = head_terminal_height.head_terminal = node
        head_terminal_height.height = 0
        while head_terminal_height.head_terminal.text is None:  # Not a terminal
            edges = [edge for edge in node.outgoing if not edge.remote and not edge.child.implicit]
            if not edges or head_terminal_height.height > MAX_HEIGHT:
                head_terminal_height.head_terminal = head_terminal_height.height = None
                break
            head_terminal_height.head_terminal = min(edges, key=lambda edge: EDGE_PRIORITY.get(edge.tag, 0)).child
            head_terminal_height.height += 1
    return head_terminal_height.height if return_height else head_terminal_height.head_terminal


def has_gaps(node, *_):  # Possibly the same as FoundationalNode.discontiguous
    return any(length > 0 for length in gap_lengths(node))


def gap_length_sum(node, *_):
    return sum(gap_lengths(node))


def gap_lengths(node, *_):
    terminals = node.get_terminals()
    return (t1.index - t2.index - 1 for (t1, t2) in zip(terminals[1:], terminals[:-1]))


def gap_type(node, *_):
    if node.text is None:  # Not a terminal
        if has_gaps(node):
            return 1  # Pass
        if any(child.text is None and has_gaps(child) for child in node.children):
            return 2  # Source
    return 0  # None


def dep_distance(node1, node2, *_):
    terminals = list(map(head_terminal, (node1, node2)))
    for v in 1, -1:
        t1, t2 = terminals[::v]
        if t1.index + t1.tok[Attr.HEAD.value] == t2.index:
            return v
    return None


def get_punctuation(nodes, terminals):
    if len(nodes) < 2:
        return None
    t0, t1 = sorted([head_terminal(node) for node in nodes], key=lambda t: t.index)
    return [t for t in terminals[t0.index:t1.index - 1] if t.tag == layer0.NodeTags.Punct]


ACTION_PROP_GETTERS = {
    "A": lambda a, *_: a.type,
    "e": lambda a, *_: a.tag if isinstance(a.tag, str) or Config().args.missing_node_features else None,  # Swap, Label
}


NODE_PROP_GETTERS = {
    "w": lambda node, *_: head_terminal(node).tok[Attr.ORTH.value],
    "m": lambda node, *_: head_terminal(node).tok[Attr.LEMMA.value],
    "t": lambda node, *_: head_terminal(node).tok[Attr.TAG.value],
    "u": lambda node, *_: head_terminal(node).tok[Attr.POS.value],
    "d": lambda node, prev, binary: dep_distance(prev, node) if binary else head_terminal(node).tok[Attr.DEP.value],
    "h": height,
    "i": lambda node, *_: head_terminal(node).index - 1,
    "j": lambda node, *_: node.index,
    "e": lambda node, prev, binary: next(e.tag for e in node.incoming if not binary or e.parent == prev),
    "n": lambda node, *_: node.label,
    "c": lambda node, *_: node.category,
    "x": lambda node, prev, binary: int(prev in node.parents) if binary else gap_type(node),
    "y": gap_length_sum,
    "P": lambda node, *_: len(node.incoming),
    "C": lambda node, *_: len(node.outgoing),
    "I": lambda node, *_: sum(1 for n in node.children if n.implicit),
    "E": lambda node, *_: sum(1 for e in node.outgoing if e.remote),
    "M": lambda node, *_: sum(1 for e in node.incoming if e.remote),
    "N": lambda node, *_: head_terminal(node).tok[Attr.ENT_IOB.value],
    "T": lambda node, *_: head_terminal(node).tok[Attr.ENT_TYPE.value],
    "#": lambda node, *_: head_terminal(node).tok[Attr.SHAPE.value],
    "^": lambda node, *_: head_terminal(node).tok[Attr.PREFIX.value],
    "$": lambda node, *_: head_terminal(node).tok[Attr.SUFFIX.value],
}


SEP_PROP_GETTERS = {
    "p": lambda nodes, terminals: get_punctuation(nodes, terminals)[0].tok[Attr.ORTH.value],
    "q": lambda nodes, terminals: len(get_punctuation(nodes, terminals)),
}


def prop_getter(prop, source=None):
    return (ACTION_PROP_GETTERS if source == "a" else SEP_PROP_GETTERS if prop in "pq" else NODE_PROP_GETTERS)[prop]


def calc(node, state, prop=None, getter=None, prev=None):
    if node is None:
        return None
    if getter is None:
        getter = prop_getter(prop)
    try:
        if prop in "pq":
            return getter(state.stack[-1:-3:-1], state.terminals)
        return getter(node, None if prev is None else prev.node, prev is not None)
    except (TypeError, AttributeError, IndexError, StopIteration):
        return None
