import re

import amrutil
from ucca import core, layer0, layer1
from ucca.convert import FormatConverter

COMMENT_PREFIX = "#"
ID_PATTERN = "#\s*::id\s+(\S+)"
TOK_PATTERN = "#\s*::(?:tok|snt)\s+(.*)"
DEP_PREFIX = ":"
TOP_DEP = "top"
INSTANCE_DEP = "instance"
INSTANCE_OF_DEP = "instance-of"
ALIGNMENT_PREFIX = "e."
ALIGNMENT_SEP = ","
NODE_DEP_ATTRIB = "dep"
TERMINAL_EDGE_TAG = "TERMINAL"


class AmrConverter(FormatConverter):
    def __init__(self):
        self.passage_id = self.amr_id = self.lines = self.tokens = self.return_amr = None

    def from_format(self, lines, passage_id, return_amr=False, **kwargs):
        del kwargs
        self.passage_id = passage_id
        self.return_amr = return_amr
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
        pending = amr.triples(rel=DEP_PREFIX + TOP_DEP)
        nodes = {}
        while pending:  # add normal nodes
            head, rel, dep = pending.pop()
            rel = rel.lstrip(DEP_PREFIX)
            dependents = amr.triples(head=dep)
            pending += dependents
            if dep in nodes:  # reentrancy
                l1.add_remote(nodes[head], rel, nodes[dep])
            else:
                node = l1.add_fnode(nodes.get(head), rel, implicit=dep not in amr.alignments() and not dependents)
                node.attrib[NODE_DEP_ATTRIB] = repr(dep)
                nodes[dep] = node
        if amr.tokens():
            reverse_alignments = [None] * len(amr.tokens())
            for k, v in amr.alignments().items():
                for i in v.lstrip(ALIGNMENT_PREFIX).split(ALIGNMENT_SEP):  # remove prefix and separate by comma
                    reverse_alignments[int(i)] = k
            for i, token in enumerate(amr.tokens()):  # add terminals
                triple = reverse_alignments[i]
                parent = l1.top_node if triple is None else nodes[triple[0] if isinstance(triple, tuple) else triple]
                parent.add(TERMINAL_EDGE_TAG, l0.add_terminal(text=token, punct=False))
        return (p, amr(alignments=False), self.amr_id) if self.return_amr else p

    def to_format(self, passage, **kwargs):
        del kwargs
        import penman
        return penman.encode(penman.Graph(list(self._to_triples(passage))))

    @staticmethod
    def _to_triples(passage):
        def _node_string(node):
            dep = node.attrib[NODE_DEP_ATTRIB]
            m = re.match("\w+\((.*)\)", dep)
            return m.group(1) if m else dep

        pending = list(passage.layer(layer1.LAYER_ID).top_node.outgoing)
        while pending:
            edge = pending.pop()
            if edge.tag != TERMINAL_EDGE_TAG:  # skip function nodes
                pending += edge.child.outgoing
                if edge.tag != TOP_DEP:  # omit top node from output
                    tag = INSTANCE_DEP if edge.tag == INSTANCE_OF_DEP else edge.tag
                    yield _node_string(edge.parent), tag, _node_string(edge.child)


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
