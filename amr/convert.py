import re

import amrutil
from ucca import core, layer0, layer1
from ucca.convert import FormatConverter
from ucca.layer1 import EdgeTags


class AmrConverter(FormatConverter):
    def __init__(self):
        self.passage_id = self.amr_id = self.lines = self.tokens = self.AMR = self.return_amr = None

    def from_format(self, lines, passage_id, return_amr, **kwargs):
        del kwargs
        self.passage_id = passage_id
        self.return_amr = return_amr
        self.lines = []
        self.amr_id = self.tokens = None
        for line in lines:
            line = line.lstrip()
            if line and line[0] != "#":
                self.lines.append(line)
                continue
            m = re.match("#\s*::id\s+(\S+)", line)
            if m:
                self.amr_id = m.group(1)
            m = re.match("#\s*::(?:tok|snt)\s+(.*)", line)
            if m:
                self.tokens = m.group(1).split()
            if self.lines:
                yield self._build_passage()
        if self.lines:
            yield self._build_passage()

    def _build_passage(self):
        amr = amrutil.parse(" ".join(self.lines), tokens=self.tokens)
        if self.return_amr:
            return amr, self.amr_id
        p = core.Passage(self.amr_id or self.passage_id)
        l0 = layer0.Layer0(p)
        l1 = layer1.Layer1(p)
        self.lines = []
        self.amr_id = self.tokens = None
        if not amr.nodes:
            return None
        pending = amr.triples(rel=":top")
        nodes = {}
        while pending:  # add normal nodes
            head, rel, dep = pending.pop()
            rel = rel.lstrip(":").rstrip("-of")  # FIXME handle -of properly
            dependents = amr.triples(head=dep)
            pending += dependents
            if dep in nodes:  # reentrancy
                l1.add_remote(nodes[head], rel, nodes[dep])
            else:
                node = l1.add_fnode(nodes.get(head), rel, implicit=dep not in amr.alignments() and not dependents)
                node.tag = repr(dep)
                nodes[dep] = node
        if amr.tokens():
            reverse_alignments = [None] * len(amr.tokens())
            for k, v in amr.alignments().items():
                for i in v.lstrip("e.").split(","):  # remove prefix and separate by comma
                    reverse_alignments[int(i)] = k
            for i, token in enumerate(amr.tokens()):  # add terminals
                triple = reverse_alignments[i]
                if triple is None:  # unaligned token
                    parent = l1.add_fnode(None, EdgeTags.Function)
                elif isinstance(triple, tuple):
                    parent = nodes[triple[0]]
                else:
                    parent = triple
                parent.add(EdgeTags.Terminal, l0.add_terminal(text=token, punct=False))
        return p

    def to_format(self, passage, **kwargs):
        del kwargs
        import penman
        return penman.encode(penman.Graph(list(self._to_triples(passage))))

    @staticmethod
    def _to_triples(passage):
        def _node_string(node):
            m = re.match("\w+\((.*)\)", node.tag)
            if m:
                return m.group(1)
            return node.tag

        pending = list(passage.layer(layer1.LAYER_ID).top_node.outgoing)
        while pending:
            edge = pending.pop()
            if edge.tag != EdgeTags.Function:  # skip function nodes
                pending += edge.child.outgoing
                if edge.tag != "top":  # do not print top node
                    yield _node_string(edge.parent), edge.tag, _node_string(edge.child)


def from_amr(lines, passage_id=None, return_amr=False, *args, **kwargs):
    """Converts from parsed text in AMR PENMAN format to a Passage object.

    :param lines: iterable of lines in AMR PENMAN format, describing a single passage.
    :param passage_id: ID to set for passage, overriding the ID from the file
    :param return_amr: return pair of (AMR object, AMR ID) rather than UCCA passage

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