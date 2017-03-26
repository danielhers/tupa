import re

import penman

from contrib import amrutil
from ucca import layer0, layer1, convert, textutil


COMMENT_PREFIX = "#"
ID_PATTERN = "#\s*::id\s+(\S+)"
TOK_PATTERN = "#\s*::(?:tok|snt)\s+(.*)"
DEP_PREFIX = ":"
TOP_DEP = ":top"
DEP_REPLACEMENT = {amrutil.INSTANCE_OF: "instance"}
IGNORED_EDGES = {"wiki"}
ALIGNMENT_PREFIX = "e."
ALIGNMENT_SEP = ","
TERMINAL_EDGE_TAG = layer1.EdgeTags.Terminal
VARIABLE_PREFIX = "v"
LABEL_PATTERN = re.compile("(\w+\(|\")(.*)(\)|\")")


class AmrConverter(convert.FormatConverter):
    def __init__(self):
        self.passage_id = self.amr_id = self.lines = self.tokens = self.nodes = self.return_amr = \
            self.remove_cycles = None

    def from_format(self, lines, passage_id, return_amr=False, remove_cycles=True, **kwargs):
        del kwargs
        self.passage_id = passage_id
        self.return_amr = return_amr
        self.remove_cycles = remove_cycles
        self.lines = []
        self.amr_id = self.tokens = None
        for line in lines:
            line = line.lstrip()
            if line:
                if line[0] != COMMENT_PREFIX:
                    self.lines.append(line)
                    continue
                m = re.match(ID_PATTERN, line)
                if m:
                    self.amr_id = m.group(1)
                else:
                    m = re.match(TOK_PATTERN, line)
                    if m:
                        self.tokens = [re.sub("@(.*)@", "\\1", t) for t in m.group(1).split()]
            if self.lines:
                yield self._build_passage()
        if self.lines:
            yield self._build_passage()

    def _build_passage(self):
        # amr = penman.decode(re.sub("~e\.[\d,]+", "", " ".join(self.lines)))
        amr = amrutil.parse(" ".join(self.lines), tokens=self.tokens)
        passage = next(convert.from_text(self.tokens, self.amr_id or self.passage_id))
        self.lines = []
        self.amr_id = self.tokens = None
        textutil.annotate(passage)
        l1 = passage.layer(layer1.LAYER_ID)
        self._build_layer1(amr, l1)
        self._build_layer0(self.align_nodes(amr), l1, passage.layer(layer0.LAYER_ID))
        self._update_implicit(l1)
        self._update_labels(l1)
        # return (passage, penman.encode(amr), self.amr_id) if self.return_amr else passage
        return (passage, amr(alignments=False), self.amr_id) if self.return_amr else passage

    def _build_layer1(self, amr, l1):
        def _reachable(x, y):  # is there a path from x to y? used to detect cycles
            q = [x]
            v = set()
            while q:
                x = q.pop(0)
                if x in v:
                    continue
                v.add(x)
                if x == y:
                    return True
                q += [d for _, _, d in amr.triples(head=x)]
            return False

        top = amr.triples(rel=TOP_DEP)  # start breadth-first search from :top relation
        assert len(top) == 1, "There must be exactly one %s edge, but %d are found" % (TOP_DEP, len(top))
        _, _, root = top[0]  # init with child of TOP
        pending = amr.triples(head=root)
        self.nodes = {}  # map triples to UCCA nodes: dep gets a new node each time unless it's a variable
        variables = {root: l1.top_node}  # map AMR variables to UCCA nodes
        visited = set()  # to avoid cycles
        while pending:  # breadth-first search creating layer 1 nodes
            triple = pending.pop(0)
            if triple in visited:
                continue
            visited.add(triple)
            head, rel, dep = triple
            rel = rel.lstrip(DEP_PREFIX)
            if rel in IGNORED_EDGES:
                continue
            parent = variables.get(head)
            assert parent is not None, "Outgoing edge from a non-variable: " + str(triple)
            node = variables.get(dep)
            if node is None:  # first occurrence of dep, or dep is not a variable
                pending += amr.triples(head=dep)  # to continue breadth-first search
                node = l1.add_fnode(parent, rel)
                if isinstance(dep, amrutil.amr_lib.Var):
                    variables[dep] = node
                else:  # save concept name / constant value in node attributes
                    node.attrib[amrutil.LABEL_ATTRIB] = repr(dep)
            elif not self.remove_cycles or not _reachable(dep, head):  # reentrancy; do not add if results in a cycle
                l1.add_remote(parent, rel, node)
            self.nodes[triple] = node

    @staticmethod
    def _build_layer0(reverse_alignments, l1, l0, ):  # add edges to terminals according to alignments
        for i, parents in reverse_alignments.items():
            terminal = l0.all[i]
            parents[0].add(TERMINAL_EDGE_TAG, terminal)
            for parent in parents[1:]:  # add as remote terminal child to all parents but the first
                if parent not in terminal.parents:  # avoid multiple identical edges (e.g. :polarity~e.68 -~e.68)
                    l1.add_remote(parent, TERMINAL_EDGE_TAG, terminal)

    def align_nodes(self, amr):
        reverse_alignments = {}
        for triple, align in {**amr.alignments(), **amr.role_alignments()}.items():
            node = self.nodes.get(triple)  # add relation alignments to dependent node
            if node is not None:  # it might be none if it was part of a removed cycle
                for i in align.lstrip(ALIGNMENT_PREFIX).split(ALIGNMENT_SEP):  # separate strings to numeric indices
                    reverse_alignments.setdefault(int(i), []).append(node)
        return reverse_alignments

    @staticmethod
    def _update_implicit(l1):
        # set implicit attribute for nodes with no terminal descendants
        pending = [n for n in l1.all if not n.children]
        while pending:
            node = pending.pop(0)
            if any(n in pending for n in node.children):
                pending.append(node)
            elif all(n.attrib.get("implicit") for n in node.children):
                node.attrib["implicit"] = True
                pending += node.parents

    @staticmethod
    def _update_labels(l1):
        for node in l1.all:
            node.attrib[amrutil.LABEL_ATTRIB] = AmrConverter.resolve_label(node, reverse=True)

    def to_format(self, passage, **kwargs):
        del kwargs
        textutil.annotate(passage)
        return penman.encode(penman.Graph(list(self._to_triples(passage)))),

    @staticmethod
    def _to_triples(passage):
        class _IdGenerator:
            def __init__(self):
                self._id = 0

            def __call__(self):
                self._id += 1
                return self._id

        def _node_label(node):
            label = labels.setdefault(node.ID, AmrConverter.resolve_label(node) or "%s%d" % (VARIABLE_PREFIX, id_gen()))
            m = re.match("\w+\((.*)\)", label)
            return m.group(1) if m else label

        id_gen = _IdGenerator()
        pending = list(passage.layer(layer1.LAYER_ID).top_node)
        visited = set()  # to avoid cycles
        labels = {}
        while pending:
            edge = pending.pop(0)
            if edge not in visited and edge.tag != TERMINAL_EDGE_TAG:  # skip cycles and terminals
                visited.add(edge)
                pending += edge.child
                tag = DEP_REPLACEMENT.get(edge.tag, edge.tag)
                yield _node_label(edge.parent), tag, _node_label(edge.child)

    @staticmethod
    def resolve_label(node, label=None, reverse=False):
        def _replace():  # replace only inside the label value/name
            m = LABEL_PATTERN.match(label)
            return (m.group(1) + m.group(2).replace(old, new) + m.group(3)) if m else label.replace(old, new)

        if label is None:
            try:
                label = node.label
            except AttributeError:
                label = node.attrib.get(amrutil.LABEL_ATTRIB)
        if label is None:
            return None
        for child in node.children:
            try:
                text = child.text
            except AttributeError:
                continue  # if it doesn't have a text attribute then it won't have a lemma attribute
            old, new = (text, amrutil.TEXT_PLACEHOLDER) if reverse else (amrutil.TEXT_PLACEHOLDER, text)
            if text and old in label:
                return _replace()
            try:
                lemma = child.lemma
            except AttributeError:
                lemma = child.extra.get(textutil.LEMMA_KEY)
            old, new = (lemma, amrutil.LEMMA_PLACEHOLDER) if reverse else (amrutil.LEMMA_PLACEHOLDER, lemma)
            if lemma and old in label:
                return _replace()
        # TODO generalize to multiple terminals: <TEXT>_<TEXT>_<TEXT> or <TEXT>/<TEXT> etc.
        return label


def from_amr(lines, passage_id=None, return_amr=False, *args, **kwargs):
    """Converts from parsed text in AMR PENMAN format to a Passage object.

    :param lines: iterable of lines in AMR PENMAN format, describing a single passage.
    :param passage_id: ID to set for passage, overriding the ID from the file
    :param return_amr: return triple of (UCCA passage, AMR string, AMR ID)

    :return generator of Passage objects
    """
    del args, kwargs
    return AmrConverter().from_format(lines, passage_id, return_amr)


def to_amr(passage, *args, **kwargs):
    """ Convert from a Passage object to a string in AMR PENMAN format (export)

    :param passage: the Passage object to convert

    :return list of lines representing an AMR in PENMAN format, constructed from the passage
    """
    del args, kwargs
    return AmrConverter().to_format(passage)


CONVERTERS = dict(convert.CONVERTERS)
CONVERTERS["amr"] = (from_amr, to_amr)
