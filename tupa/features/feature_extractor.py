import re

from ucca import layer0
from ucca.layer1 import EdgeTags

FEATURE_ELEMENT_PATTERN = re.compile("([sba])(\d)([lruLRU]*)([wtdhencpqxyAPCIRNT]*)")
FEATURE_TEMPLATE_PATTERN = re.compile("^(%s)+$" % FEATURE_ELEMENT_PATTERN.pattern)


class FeatureTemplate(object):
    """
    A feature template in parsed form, ready to be used for value calculation
    """

    def __init__(self, name, elements):
        """
        :param name: name of the feature in the short-hand form, to be used for the dictionary
        :param elements: collection of FeatureElement objects that represent the actual feature
        """
        self.name = name
        self.suffix = name[-1]
        self.elements = elements

    def __str__(self):
        return self.name


class FeatureTemplateElement(object):
    """
    One element in the values of a feature, e.g. from one node
    """

    def __init__(self, source, index, relatives, properties):
        """
        :param source: where to take the data from:
                           s: stack nodes
                           b: buffer nodes
                           a: past actions
        :param index: non-negative integer, the index of the element in the stack, buffer or list
                           of past actions (in the case of stack and actions, indexing from the end)
        :param relatives: string in [lruLRU]*, to select a descendant/parent of the node instead:
                           l: leftmost child
                           r: rightmost child
                           u: only child, if there is just one
                           L: leftmost parent
                           R: rightmost parent
                           U: only parent, if there is just one
        :param properties: the actual values to choose, if available (else omit feature), out of:
                           w: node text
                           t: node POS tag
                           d: node dependency relation
                           h: node height
                           e: tag of first incoming edge / action tag
                           n: node label
                           c: node label category suffix
                           ,: unique separator punctuation between nodes
                           q: count of any separator punctuation between nodes
                           x: numeric value of gap type
                           y: sum of gap lengths
                           A: action type label
                           P: number of parents
                           C: number of children
                           I: number of implicit children
                           R: number of remote children
                           N: numeric value of named entity IOB
                           T: named entity type
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
        self.properties = properties

    def __str__(self):
        return self.source + str(self.index) + self.relatives + self.properties

    def __eq__(self, other):
        return self.source == other.source and self.index == other.index and self.relatives == other.relatives


class FeatureExtractor(object):
    """
    Object to extract features from the parser state to be used in action classification
    """

    def __init__(self, feature_templates=(), feature_extractor=None, params=None):
        assert all(FEATURE_TEMPLATE_PATTERN.match(f) for f in feature_templates), \
            "Features do not match pattern: " + ", ".join(
                f for f in feature_templates if not FEATURE_TEMPLATE_PATTERN.match(f))
        # convert the list of features textual descriptions to the actual fields
        self.feature_templates = [FeatureTemplate(
            feature_name, tuple(FeatureTemplateElement(*m.group(1, 2, 3, 4))
                                for m in re.finditer(FEATURE_ELEMENT_PATTERN, feature_name)))
            for feature_name in feature_templates]
        self.feature_extractor = feature_extractor
        self.params = {} if params is None else params

    def extract_features(self, state, params):
        """
        Calculate feature values according to current state
        :param state: current state of the parser
        :param params: dict of FeatureParameters for each suffix
        """
        raise NotImplementedError()

    def init_features(self, state, suffix=None):
        """
        Calculate feature values for initial state
        :param state: initial state of the parser
        :param suffix: feature suffix to get
        """
        pass

    def collapse_features(self, suffixes):
        """
        For each set of features referring to the same node, with the given properties,
         leave only one of them.
        """
        pass

    def finalize(self):
        return self

    def restore(self):
        pass

    def save(self, filename):
        pass

    def load(self, filename):
        pass

    @staticmethod
    def calc_feature(feature_template, state, default=None, indexed=False):
        values = []
        prev_elem = None
        prev_node = None
        for element in feature_template.elements:
            node = get_node(element, state)
            if not element.properties:
                if prev_elem is not None:
                    assert node is None or prev_node is None, "Property-less elements: %s%s" % (prev_elem, element)
                    if default is None:
                        return None
                    values.append(default)
                prev_elem = element
                prev_node = node
            else:
                for prop in ("i",) if indexed else element.properties:
                    value = FeatureExtractor.get_prop(element, node, prev_node, prev_elem, prop, state)
                    if value is None:
                        if default is None:
                            return None
                        values.append(default)
                    else:
                        values.append(value)
                prev_elem = None
                prev_node = None
        return values

    @staticmethod
    def get_prop(element, node, prev_node, prev_elem, prop, state):
        try:
            if element is not None and element.source == "a":
                return action_prop(node, prop)
            elif prop in "pq":
                return separator_prop(state.stack[-1:-3:-1], state.terminals, prop)
            if node is None:
                return None
            return node_prop(node, prop, prev_node, prev_elem)
        except (AttributeError, StopIteration):
            return None


def get_node(element, state):
    if element.source == "s":
        if len(state.stack) <= element.index:
            return None
        node = state.stack[-1 - element.index]
    elif element.source == "b":
        if len(state.buffer) <= element.index:
            return None
        node = state.buffer[element.index]
    else:  # source == "a"
        if len(state.actions) <= element.index:
            return None
        node = state.actions[-1 - element.index]
    for relative in element.relatives:
        nodes = node.parents if relative.isupper() else node.children
        lower = relative.lower()
        if not nodes:
            return None
        elif len(nodes) == 1:
            if lower == "u":
                node = nodes[0]
        elif lower == "l":
            node = nodes[0]
        elif lower == "r":
            node = nodes[-1]
        else:  # lower == "u" and len(nodes) > 1
            return None
    return node

ACTION_PROPS = {
    "A": "type",
    "e": "tag",
}


def action_prop(action, prop):
    return getattr(action, ACTION_PROPS[prop])


def separator_prop(nodes, terminals, prop):
    if len(nodes) < 2:
        return None
    t0, t1 = sorted([head_terminal(node) for node in nodes], key=lambda t: t.index)
    punctuation = [t for t in terminals[t0.index:t1.index - 1] if t.tag == layer0.NodeTags.Punct]
    if prop == "p" and len(punctuation) == 1:
        return punctuation[0].text
    if prop == "q":
        return len(punctuation)
    return None

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
    return head_terminal_height(node, True)


def head_terminal_height(node, return_height=False):
    if node is not head_terminal_height.node:
        head_terminal_height.node = head_terminal_height.head_terminal = node
        head_terminal_height.height = 0
        while head_terminal_height.head_terminal.text is None:  # Not a terminal
            edges = [edge for edge in node.outgoing if not edge.remote and not edge.child.implicit]
            if not edges or head_terminal_height.height > 30:
                head_terminal_height.head_terminal = head_terminal_height.height = None
                break
            head_terminal_height.head_terminal = min(edges, key=lambda edge: EDGE_PRIORITY.get(edge.tag, 0)).child
            head_terminal_height.height += 1
    return head_terminal_height.height if return_height else head_terminal_height.head_terminal
head_terminal_height.node = head_terminal_height.head_terminal = head_terminal_height.height = None


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


def dependency_distance(node1, node2, *_):
    t1, t2 = head_terminal(node1), head_terminal(node2)
    if t1.dep_head == t2.index:
        return 1
    elif t2.dep_head == t1.index:
        return -1
    return None

NODE_PROP_GETTERS = {
    "w": lambda node, *_: head_terminal(node).text,
    "t": lambda node, *_: head_terminal(node).pos_tag,
    "d": lambda node, prev, binary: dependency_distance(prev, node) if binary else head_terminal(node).dep_rel,
    "h": height,
    "i": lambda node, *_: head_terminal(node).index - 1,
    "e": lambda node, prev, binary: next(e.tag for e in node.incoming if not binary or e.parent == prev),
    "n": lambda node, *_: node.label,
    "c": lambda node, *_: node.category,
    "x": lambda node, prev, binary: int(prev in node.parents) if binary else gap_type(node),
    "y": gap_length_sum,
    "P": lambda node, *_: len(node.incoming),
    "C": lambda node, *_: len(node.outgoing),
    "I": lambda node, *_: len([n for n in node.children if n.implicit]),
    "R": lambda node, *_: len([e for e in node.outgoing if e.remote]),
    "N": lambda node, *_: int(head_terminal(node).ner_iob),
    "T": lambda node, *_: head_terminal(node).ner_type,
}


def node_prop(node, prop, prev_node, prev_elem):
    return NODE_PROP_GETTERS[prop](node, prev_node, prev_elem is not None)
