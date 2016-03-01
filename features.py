import re

from ucca import layer0
from ucca.layer1 import EdgeTags

FEATURE_ELEMENT_PATTERN = re.compile("([sba])(\d)([lrup]*)([wtepqxyPCIR]*)")
FEATURE_TEMPLATE_PATTERN = re.compile("^(%s)+$" % FEATURE_ELEMENT_PATTERN.pattern)

FEATURE_TEMPLATES = (
    # unigrams (Zhang and Clark 2009):
    "s0te", "s0we", "s1te", "s1we", "s2te", "s2we", "s3te", "s3we",
    "b0wt", "b1wt", "b2wt", "b3wt",
    "s0lwe", "s0rwe", "s0uwe", "s1lwe", "s1rwe", "s1uwe",
    # bigrams (Zhang and Clark 2009):
    "s0ws1w", "s0ws1e", "s0es1w", "s0es1e", "s0wb0w", "s0wb0t",
    "s0eb0w", "s0eb0t", "s1wb0w", "s1wb0t", "s1eb0w", "s1eb0t",
    "b0wb1w", "b0wb1t", "b0tb1w", "b0tb1t",
    # trigrams (Zhang and Clark 2009):
    "s0es1es2w", "s0es1es2e", "s0es1es2e", "s0es1eb0w", "s0es1eb0t",
    "s0es1wb0w", "s0es1wb0t", "s0ws1es2e", "s0ws1eb0t",
    # extended (Zhu et al. 2013):
    "s0llwe", "s0lrwe", "s0luwe", "s0rlwe", "s0rrwe",
    "s0ruwe", "s0ulwe", "s0urwe", "s0uuwe", "s1llwe",
    "s1lrwe", "s1luwe", "s1rlwe", "s1rrwe", "s1ruwe",
    # parents:
    "s0pwe", "s1pwe", "b0pwe",
    # separator (Zhu et al. 2013):
    "s0wp", "s0wep", "s0wq", "s0weq", "s0es1ep", "s0es1eq",
    "s1wp", "s1wep", "s1wq", "s1weq",
    # disco, unigrams (Maier 2015):
    "s0xwe", "s1xwe", "s2xwe", "s3xwe",
    "s0xte", "s1xte", "s2xte", "s3xte",
    "s0xy", "s1xy", "s2xy", "s3xy",
    # disco, bigrams (Maier 2015):
    "s0xs1e", "s0xs1w", "s0xs1x", "s0ws1x", "s0es1x",
    "s0xs2e", "s0xs2w", "s0xs2x", "s0es2x",
    "s0ys1y", "s0ys2y", "s0xb0t", "s0xb0w",
    # counts (Tokgöz and Eryiğit 2015):
    "s0P", "s0C", "s0wP", "s0wC",
    "b0P", "b0C", "b0wP", "b0wC",
    # existing edges (Tokgöz and Eryiğit 2015):
    "s0s1", "s1s0", "s0b0", "b0s0",
    "s0b0e", "b0s0e",
    # past actions (Tokgöz and Eryiğit 2015):
    "a0we", "a1we",
    # UCCA-specific
    "s0I", "s0R", "s0wI", "s0wR",
    "b0I", "b0R", "b0wI", "b0wR",
)


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
        self.elements = elements


class FeatureTemplateElement(object):
    """
    One element in the values of a feature, e.g. from one node
    """
    def __init__(self, source, index, children, properties):
        """
        :param source: where to take the data from:
                           s: stack nodes
                           b: buffer nodes
                           a: past actions
        :param index: non-negative integer, the index of the element in the stack, buffer or list
                           of past actions (in the case of stack and actions, indexing from the end)
        :param children: string in [lrup]*, to select a descendant of the node instead:
                           l: leftmost child
                           r: rightmost child
                           u: only child, if there is just one
                           p: parent
        :param properties: the actual values to choose, if available (else omit feature), out of:
                           w: node text / action type
                           t: node POS tag
                           e: tag of first incoming edge / action tag
                           p: unique separator punctuation between nodes
                           q: count of any separator punctuation between nodes
                           x: gap type
                           y: sum of gap lengths
                           P: number of parents
                           C: number of children
                           I: number of implicit children
                           R: number of remote children
                           If empty, the value will be 1 if there is an edge from this node to the
                           next one in the template, or 0 otherwise. Also, if the next node comes
                           with the "e" property, then the edge with this node will be considered.
        """
        self.source = source
        self.index = int(index)
        self.children = children
        self.properties = properties


class FeatureExtractor(object):
    """
    Object to extract features from the parser state to be used in action classification
    """
    def __init__(self):
        assert all(FEATURE_TEMPLATE_PATTERN.match(f) for f in FEATURE_TEMPLATES),\
            "Features do not match pattern: " + ", ".join(f for f in FEATURE_TEMPLATES
                                                          if not FEATURE_TEMPLATE_PATTERN.match(f))
        # convert the list of features textual descriptions to the actual fields
        self.feature_templates = [FeatureTemplate(feature_name,
                                                  tuple(FeatureTemplateElement(*m.group(1, 2, 3, 4))
                                                        for m in re.finditer(FEATURE_ELEMENT_PATTERN,
                                                                             feature_name)))
                                  for feature_name in FEATURE_TEMPLATES]

    def extract_features(self, state):
        """
        Calculate feature values according to current state
        :param state: current state of the parser
        """
        features = {
            "b": 1,  # Bias
            "n/t": state.node_ratio(),  # number of non-terminals divided by number of terminals
        }
        for feature_template in self.feature_templates:
            values = calc_feature(feature_template, state)
            if values is not None:
                features["%s=%s" % (feature_template.name, " ".join(values))] = 1
        return features


def calc_feature(feature_template, state):
    values = []
    prev_node = None
    for element in feature_template.elements:
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
        for child in element.children:
            if child == "p":
                if node.parents:
                    node = node.parents[0]
                else:
                    return None
            elif not node.children:
                return None
            elif len(node.children) == 1:
                if child == "u":
                    node = node.children[0]
            elif child == "l":
                node = node.children[0]
            elif child == "r":
                node = node.children[-1]
            else:  # child == "u" and len(node.children) > 1
                return None
        if not element.properties:
            if prev_node is not None:
                values.append("1" if prev_node in node.parents else "0")
            prev_node = node
        for p in element.properties:
            try:
                if element.source == "a":
                    v = get_action_prop(node, p)
                elif p in "pq":
                    v = get_separator_prop(state.stack[-1:-3:-1], state.terminals, p)
                else:
                    v = get_prop(node, p, prev_node)
            except (AttributeError, StopIteration):
                v = None
            if v is None:
                return None
            values.append(str(v))
    return values


def get_prop(node, p, prev_node=None):
    if p == "w":
        return get_head_terminal(node).text
    if p == "t":
        return get_head_terminal(node).pos_tag
    if p == "e":
        return next(e.tag for e in node.incoming if prev_node is None or e.parent == prev_node)
    if p == "x":
        return gap_type(node)
    if p == "y":
        return gap_length_sum(node)
    if p == "P":
        return len(node.incoming)
    if p == "C":
        return len(node.outgoing)
    if p == "I":
        return len([n for n in node.children if n.implicit])
    if p == "R":
        return len([e for e in node.outgoing if e.remote])
    raise Exception("Unknown node property: " + p)


def get_action_prop(action, p):
    if p == "w":
        return action.type
    if p == "e":
        return action.tag
    raise Exception("Unknown action property: " + p)


def get_separator_prop(nodes, terminals, p):
    if len(nodes) < 2:
        return None
    t0, t1 = sorted([get_head_terminal(node) for node in nodes], key=lambda t: t.index)
    punctuation = [terminal for terminal in terminals[t0.index + 1:t1.index]
                   if terminal.tag == layer0.NodeTags.Punct]
    if p == "p" and len(punctuation) == 1:
        return punctuation[0].text
    if p == "q":
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


def get_head_terminal(node):
    while node.text is None:  # Not a terminal
        edges = [edge for edge in node.outgoing
                 if not edge.remote and not edge.child.implicit]
        if not edges:
            return None
        node = min(edges, key=lambda edge: EDGE_PRIORITY.get(edge.tag, 0)).child
    return node


def has_gaps(node):
    # Possibly the same as FoundationalNode.discontiguous
    return any(length > 0 for length in gap_lengths(node))


def gap_length_sum(node):
    return sum(gap_lengths(node))


def gap_lengths(node):
    terminals = node.get_terminals()
    return (t1.index - t2.index - 1 for (t1, t2) in zip(terminals[1:], terminals[:-1]))


def gap_type(node):
    if node.text is not None:
        return "n"  # None
    if has_gaps(node):
        return "p"  # Pass
    if any(child.text is None and has_gaps(child) for child in node.children):
        return "s"  # Source
    return "n"  # None
