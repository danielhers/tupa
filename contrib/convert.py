import re

from contrib import amrutil
from ucca import core, layer0, layer1, convert

COMMENT_PREFIX = "#"
ID_PATTERN = "#\s*::id\s+(\S+)"
TOK_PATTERN = "#\s*::(?:tok|snt)\s+(.*)"
DEP_PREFIX = ":"
TOP_DEP = "top"
INSTANCE_DEP = "instance"
INSTANCE_OF_DEP = "instance-of"
ALIGNMENT_PREFIX = "e."
ALIGNMENT_SEP = ","
NODE_NAME_ATTRIB = "name"
TERMINAL_EDGE_TAG = layer1.EdgeTags.Terminal


class AmrConverter(convert.FormatConverter):
    def __init__(self):
        self.passage_id = self.amr_id = self.lines = self.tokens = self.return_amr = self.remove_cycles = None

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
                        self.tokens = m.group(1).split()
            if self.lines:
                yield self._build_passage()
        if self.lines:
            yield self._build_passage()

    def _build_passage(self):
        amr = amrutil.parse(" ".join(self.lines), tokens=self.tokens)
        p = core.Passage(self.amr_id or self.passage_id)
        l0 = layer0.Layer0(p)
        l1 = layer1.Layer1(p)
        self.lines = []
        self.amr_id = self.tokens = None
        nodes = self._build_layer1(amr, l1, self.remove_cycles)
        self._build_layer0(amr, l0, l1, nodes)
        self._update_implicit(l1)
        return (p, amr(alignments=False), self.amr_id) if self.return_amr else p

    @staticmethod
    def _build_layer1(amr, l1, remove_cycles):
        def _reachable(x, y):
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

        nodes = {}
        visited = set()  # to avoid cycles
        pending = amr.triples(rel=DEP_PREFIX + TOP_DEP)
        while pending:  # add normal nodes
            triple = pending.pop(0)
            if triple in visited:
                continue
            visited.add(triple)
            head, rel, dep = triple
            rel = rel.lstrip(DEP_PREFIX)
            if dep in nodes:  # reentrancy
                if not remove_cycles or not _reachable(dep, head):
                    l1.add_remote(nodes[head], rel, nodes[dep])
            else:
                pending += amr.triples(head=dep)
                node = l1.add_fnode(nodes.get(head), rel)
                node.attrib[NODE_NAME_ATTRIB] = repr(dep)
                nodes[dep] = node
        return nodes

    @staticmethod
    def _build_layer0(amr, l0, l1, nodes):  # add edges to terminals according to alignments
        reverse_alignments = {}
        for k, v in amr.alignments().items():
            node = nodes.get(k[2] if isinstance(k, tuple) else k)  # change relation alignments to the dependent instead
            if node is not None:  # it might be none if it was part of a removed cycle
                for i in v.lstrip(ALIGNMENT_PREFIX).split(ALIGNMENT_SEP):  # separate strings to numeric indices
                    reverse_alignments.setdefault(int(i), []).append(node)
        for i, token in enumerate(amr.tokens()):  # add terminals, unaligned tokens will be the root's children
            terminal = l0.add_terminal(text=token, punct=False)
            parents = reverse_alignments.get(i, [l1.top_node])
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

    def to_format(self, passage, **kwargs):
        del kwargs
        import penman
        return penman.encode(penman.Graph(list(self._to_triples(passage)))),

    @staticmethod
    def _to_triples(passage):
        class _IdGenerator:
            def __init__(self):
                self._id = 0

            def __call__(self):
                self._id += 1
                return self._id

        def _node_name(node):
            name = node.attrib.get(NODE_NAME_ATTRIB, "v%d" % id_generator())
            m = re.match("\w+\((.*)\)", name)
            return m.group(1) if m else name

        id_generator = _IdGenerator()
        pending = list(passage.layer(layer1.LAYER_ID).top_node.outgoing)
        visited = set()  # to avoid cycles
        while pending:
            edge = pending.pop(0)
            if edge not in visited and edge.tag != TERMINAL_EDGE_TAG:  # skip cycles and unaligned terminals
                visited.add(edge)
                pending += edge.child.outgoing
                if edge.tag != TOP_DEP:  # omit top node from output
                    tag = INSTANCE_DEP if edge.tag == INSTANCE_OF_DEP else edge.tag
                    yield _node_name(edge.parent), tag, _node_name(edge.child)


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
