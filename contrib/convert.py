import re

import penman

from contrib import amrutil
from ucca import core, layer0, layer1, convert, textutil


COMMENT_PREFIX = "#"
ID_PATTERN = "#\s*::id\s+(\S+)"
TOK_PATTERN = "#\s*::(?:tok|snt)\s+(.*)"
DEP_PREFIX = ":"
TOP_DEP = "top"
DEP_REPLACEMENT = {amrutil.INSTANCE_OF: "instance"}
IGNORED_EDGES = {"wiki"}
ALIGNMENT_PREFIX = "e."
ALIGNMENT_SEP = ","
TERMINAL_EDGE_TAG = layer1.EdgeTags.Terminal
VARIABLE_PREFIX = "v"


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
        p = core.Passage(self.amr_id or self.passage_id)
        self.lines = []
        self.amr_id = self.tokens = None
        l1 = layer1.Layer1(p)
        self._build_layer1(amr, l1)
        self._build_layer0(amr, layer0.Layer0(p), l1)
        self._update_implicit(l1)
        textutil.annotate(p)
        self._update_label(l1)
        # return (p, penman.encode(amr), self.amr_id) if self.return_amr else p
        return (p, amr(alignments=False), self.amr_id) if self.return_amr else p

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

        self.nodes = {}
        visited = set()  # to avoid cycles
        pending = amr.triples(rel=DEP_PREFIX + TOP_DEP)
        while pending:  # add normal nodes
            triple = pending.pop(0)
            if triple in visited:
                continue
            visited.add(triple)
            head, rel, dep = triple
            rel = rel.lstrip(DEP_PREFIX)
            if rel in IGNORED_EDGES:
                continue
            node = self.nodes.get(dep)
            if node is None:  # first occurrence of dep, or not a variable
                pending += amr.triples(head=dep)
                node = l1.top_node if rel == TOP_DEP else l1.add_fnode(self.nodes[head], rel)
                self.nodes[dep] = node
                if not isinstance(dep, amrutil.amr_lib.Var):
                    node.attrib[amrutil.LABEL_ATTRIB] = repr(dep)
            elif not self.remove_cycles or not _reachable(dep, head):  # reentrancy; do not add if results in a cycle
                l1.add_remote(self.nodes[head], rel, node)

    def _build_layer0(self, amr, l0, l1):  # add edges to terminals according to alignments
        reverse_alignments = {}
        for (h, r, d), v in {**amr.alignments(), **amr.role_alignments()}.items():
            node = self.nodes.get(d)  # change relation alignments to dependent
            if node is not None:  # it might be none if it was part of a removed cycle
                for i in v.lstrip(ALIGNMENT_PREFIX).split(ALIGNMENT_SEP):  # separate strings to numeric indices
                    reverse_alignments.setdefault(int(i), []).append(node)
        for i, token in enumerate(amr.tokens()):  # add terminals, unaligned tokens will be the root's children
            terminal = l0.add_terminal(text=token, punct=False)
            parents = reverse_alignments.get(i)
            if parents:
                parents[0].add(TERMINAL_EDGE_TAG, terminal)
                for parent in parents[1:]:  # add as remote terminal child to all parents but the first
                    if parent not in terminal.parents:  # avoid multiple identical edges (e.g. :polarity~e.68 -~e.68)
                        l1.add_remote(parent, TERMINAL_EDGE_TAG, terminal)

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
    def _update_label(l1):
        for node in l1.all:
            node.attrib[amrutil.LABEL_ATTRIB] = amrutil.resolve_label(node, reverse=True)

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
            label = labels.setdefault(node.ID, amrutil.resolve_label(node) or "%s%d" % (VARIABLE_PREFIX, id_gen()))
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
