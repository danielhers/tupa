import re

import penman
from nltk.corpus import wordnet as wn

from contrib import amrutil
from ucca import layer0, layer1, convert, textutil


COMMENT_PREFIX = "#"
ID_PATTERN = "#\s*::id\s+(\S+)"
TOK_PATTERN = "#\s*::(?:tok|snt)\s+(.*)"
DEP_PREFIX = ":"
TOP_DEP = ":top"
DEP_REPLACEMENT = {amrutil.INSTANCE_OF: "instance"}
IGNORED_AMR_EDGES = {"wiki"}
ALIGNMENT_PREFIX = "e."
ALIGNMENT_SEP = ","


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
                        self.tokens = [t.strip("@") for t in re.sub("\\\\|(?<=<)[^<>]+(?=>)", "", m.group(1)).split()]
            if self.lines:
                yield self._build_passage()
        if self.lines:
            yield self._build_passage()

    def _build_passage(self):
        # amr = penman.decode(re.sub("~e\.[\d,]+", "", " ".join(self.lines)))
        amr = amrutil.parse(" ".join(self.lines), tokens=self.tokens)
        passage = next(convert.from_text(self.tokens, self.amr_id or self.passage_id, tokenized=True))
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
        l1.top_node.attrib[amrutil.LABEL_ATTRIB] = amrutil.VARIABLE_LABEL  # the root is always a variable
        variables = {root: l1.top_node}  # map AMR variables to UCCA nodes
        visited = set()  # to avoid cycles
        while pending:  # breadth-first search creating layer 1 nodes
            triple = pending.pop(0)
            if triple in visited:
                continue
            visited.add(triple)
            head, rel, dep = triple
            rel = rel.lstrip(DEP_PREFIX)
            if rel in IGNORED_AMR_EDGES:
                continue
            parent = variables.get(head)
            assert parent is not None, "Outgoing edge from a non-variable: " + str(triple)
            node = variables.get(dep)
            if node is None:  # first occurrence of dep, or dep is not a variable
                pending += amr.triples(head=dep)  # to continue breadth-first search
                node = l1.add_fnode(parent, rel)
                if isinstance(dep, amrutil.amr_lib.Var):
                    variables[dep] = node
                    label = amrutil.VARIABLE_LABEL
                else:  # save concept name / constant value in node attributes
                    label = repr(dep)
                node.attrib[amrutil.LABEL_ATTRIB] = label
            elif not self.remove_cycles or not _reachable(dep, head):  # reentrancy; do not add if results in a cycle
                l1.add_remote(parent, rel, node)
            self.nodes[triple] = node

    @staticmethod
    def _build_layer0(reverse_alignments, l1, l0):  # add edges to terminals according to alignments
        for i, parents in reverse_alignments.items():
            terminal = l0.all[i]
            if layer0.is_punct(terminal):
                tag = layer1.EdgeTags.Punctuation
                terminal = l1.add_punct(parents[0], terminal)
                terminal.attrib[amrutil.LABEL_ATTRIB] = layer1.NodeTags.Punctuation
            else:
                tag = layer1.EdgeTags.Terminal
                parents[0].add(tag, terminal)
            for parent in parents[1:]:  # add as remote terminal child to all parents but the first
                if parent not in terminal.parents:  # avoid multiple identical edges (e.g. :polarity~e.68 -~e.68)
                    l1.add_remote(parent, tag, terminal)

    def align_nodes(self, amr):
        reverse_alignments = {}
        tokens = amr.tokens()
        for triple, align in {**amr.alignments(), **amr.role_alignments()}.items():
            node = self.nodes.get(triple)  # add relation alignments to dependent node
            if node is not None:  # it might be none if it was part of a removed cycle
                indices = list(map(int, align.lstrip(ALIGNMENT_PREFIX).split(ALIGNMENT_SEP)))  # split numeric indices
                label = str(triple[2])  # correct missing alignment by expanding to terminals contained in label
                for start, offset in ((0, -1), (-1, 1)):
                    i = indices[start] + offset
                    while 0 <= i < len(tokens) and tokens[i] in label:
                        indices.append(i)
                        i += offset
                for i, token in enumerate(tokens):
                    if i not in indices and len(token) > 2 and \
                                    token in self.strip(label).strip('"') and tokens.count(token) == 1:
                        indices.append(i)
                for i in indices:
                    reverse_alignments.setdefault(i, []).append(node)
        return reverse_alignments

    @staticmethod
    def _update_implicit(l1):
        # set implicit attribute for nodes with no terminal descendants
        pending = [n for n in l1.all if not n.children]
        while pending:
            node = pending.pop(0)
            if node in l1.heads:
                pass
            elif any(n in pending for n in node.children):
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

            def __call__(self, label):
                if label == amrutil.VARIABLE_LABEL:
                    self._id += 1
                    return label + str(self._id)
                return label

        def _node_label(node):
            return AmrConverter.strip(labels.setdefault(node.ID, id_gen(AmrConverter.resolve_label(node))))

        id_gen = _IdGenerator()
        pending = list(passage.layer(layer1.LAYER_ID).top_node)
        visited = set()  # to avoid cycles
        labels = {}
        while pending:
            edge = pending.pop(0)
            if edge not in visited and edge.tag not in amrutil.TERMINAL_TAGS:  # skip cycles and terminals
                visited.add(edge)
                pending += edge.child
                tag = DEP_REPLACEMENT.get(edge.tag, edge.tag)
                yield _node_label(edge.parent), tag, _node_label(edge.child)

    @staticmethod
    def strip(label):
        return re.sub("\w+\((.*)\)", "\\1", label)

    @staticmethod
    def resolve_label(node, label=None, reverse=False):
        def _replace(old, new):  # replace only inside the label value/name
            new = new.strip('"()')
            if reverse:
                old, new = new, old
            replaceable = old and (len(old) > 2 or len(label) < 5)
            return re.sub(re.escape(old) + "(?![^<]*>|[^(]*\(|\d+$)", new, label) if replaceable else label

        def _related_forms(w):  # list of all derivationally related forms and their part of speech
            num_related = 0
            related = {None: w}
            while len(related) > num_related:
                num_related = len(related)
                related.update({v.synset().pos(): v.name() for x in related.values()
                                for l in wn.lemmas(x) for v in l.derivationally_related_forms()})
            return [(v, k) for k, v in related.items() if v != w]

        if label is None:
            try:
                label = node.label
            except AttributeError:
                label = node.attrib[amrutil.LABEL_ATTRIB]
        if label != amrutil.VARIABLE_LABEL:
            terminals = [c for c in node.children if getattr(c, "text", None)]
            if len(terminals) > 1:
                label = _replace("<t>", "".join(t.text for t in terminals))
            for i, terminal in enumerate(terminals):
                label = _replace("<t%d>" % i, terminal.text)
                label = _replace("<T%d>" % i, terminal.text.title())
                try:  # TODO add lemma and related forms to classifier features
                    lemma = terminal.lemma
                except AttributeError:
                    lemma = terminal.extra.get(textutil.LEMMA_KEY)
                if lemma == "-PRON-":
                    lemma = terminal.text.lower()
                label = _replace("<l%d>" % i, lemma)
                label = _replace("<L%d>" % i, lemma.title())
                for form, pos in _related_forms(lemma):
                    label = _replace("<%s%d>" % (pos, i), form)
        return label


# REPLACEMENTS = {"~": "about"}


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
