from collections import namedtuple, OrderedDict

import penman
from ucca import layer0, convert

from ..util.amr import *

DELETE_PATTERN = re.compile("\\\\|(?<=(?<!<)<)[^<>]+(?=>(?!>))")  # Delete text inside single angle brackets


class AmrConverter(convert.FormatConverter):
    def __init__(self):
        self.passage_id = self.amr_id = self.lines = self.tokens = self.nodes = self.return_original = \
            self.remove_cycles = self.extensions = self.excluded = None

    def from_format(self, lines, passage_id, return_original=False, remove_cycles=True, **kwargs):
        self.passage_id = passage_id
        self.return_original = return_original
        self.remove_cycles = remove_cycles
        self.extensions = [l for l in EXTENSIONS if kwargs.get(l)]
        self.excluded = {i for l, r in EXTENSIONS.items() if l not in self.extensions for i in r}
        self.lines = []
        self.amr_id = self.tokens = None
        for line in lines:
            line = line.lstrip()
            if line:
                if line[0] != COMMENT_PREFIX:
                    self.lines.append(line)
                    continue
                m = ID_PATTERN.match(line)
                if m:
                    self.amr_id = m.group(1)
                else:
                    m = TOK_PATTERN.match(line)
                    if m:
                        self.tokens = [t.strip("@") or "@" for t in DELETE_PATTERN.sub("", m.group(1)).split()]
            if self.lines:
                yield self._build_passage()
        if self.lines:
            yield self._build_passage()

    def _build_passage(self):
        assert self.tokens is not None, "Cannot convert AMR without input tokens"
        amr = parse(" ".join(self.lines), tokens=self.tokens)
        amr_id = self.amr_id or self.passage_id
        passage = next(convert.from_text(self.tokens, amr_id, tokenized=True))
        passage.extra["format"] = "amr"
        self.lines = []
        self.amr_id = self.tokens = None
        textutil.annotate(passage)
        l0 = passage.layer(layer0.LAYER_ID)
        l1 = passage.layer(layer1.LAYER_ID)
        self._build_layer1(amr, l1)
        self._build_layer0(self.align_nodes(amr), l1, l0)
        self._update_implicit(l1)
        self._update_labels(l1)
        return (passage, amr(alignments=False), amr_id) if self.return_original else passage

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
        self.nodes = OrderedDict()  # map triples to UCCA nodes: dep gets a new node each time unless it's a variable
        variables = {root: l1.heads[0]}  # map AMR variables to UCCA nodes
        names = set()  # to collapse :name (... / name) :op "..." into one string node
        excluded = set()  # nodes whose outgoing edges (except for instance-of edges) will be ignored
        visited = set()  # to avoid cycles
        while pending:  # breadth-first search creating layer 1 nodes
            triple = pending.pop(0)
            if triple in visited:
                continue
            visited.add(triple)
            head, rel, dep = triple
            if rel in self.excluded or head in excluded:
                continue  # skip edges whose relation belongs to excluded layers
            if dep in self.excluded:
                excluded.add(head)  # skip outgoing edges from variables with excluded concepts
            rel = rel.lstrip(DEP_PREFIX)  # remove : prefix
            rel = PREFIXED_RELATION_PATTERN.sub(PREFIXED_RELATION_SUBSTITUTION, rel)  # remove numeric/prep suffix
            if rel == NAME:
                names.add(dep)
            parent = variables.get(head)
            assert parent is not None, "Outgoing edge from a non-variable: " + str(triple)
            node = variables.get(dep)
            if node is None:  # first occurrence of dep, or dep is not a variable
                pending += amr.triples(head=dep)  # to continue breadth-first search
                dep_is_concept = isinstance(dep, amr_lib.Concept)
                head_is_name = head in names
                node = parent if dep_is_concept or head_is_name else l1.add_fnode(parent, rel)
                dep_str = repr(dep)
                if isinstance(dep, amr_lib.Var):
                    variables[dep] = node
                elif head_is_name and (dep_is_concept or rel == OP):  # collapse name ops to one string node
                    if not dep_is_concept:  # the instance-of relation is dropped
                        label = node.attrib.get(LABEL_ATTRIB)
                        node.attrib[LABEL_ATTRIB] = '"%s"' % "_".join(
                            AmrConverter.strip(l, strip_quotes=True) for l in (label, dep_str) if l)
                else:  # concept or constant: save value in node attributes
                    node.attrib[LABEL_ATTRIB] = dep_str  # concepts are saved as variable labels, not as actual nodes
            elif not self.remove_cycles or not _reachable(dep, head):  # reentrancy; do not add if results in a cycle
                l1.add_remote(parent, rel, node)
            self.nodes[triple] = node

    @staticmethod
    def _build_layer0(preterminals, l1, l0):  # add edges to terminals according to alignments
        for i, parents in preterminals.items():
            terminal = l0.all[i]
            if layer0.is_punct(terminal):
                tag = PUNCTUATION_DEP
                terminal = l1.add_punct(parents[0], terminal)
                terminal.attrib[LABEL_ATTRIB] = PUNCTUATION_LABEL
                del parents[1:]  # avoid multiple punctuation parents, which is mostly due to alignment errors
            else:
                tag = TERMINAL_DEP
            for parent in parents:
                if parent not in terminal.parents:  # avoid multiple identical edges (e.g. :polarity~e.68 -~e.68)
                    parent.add(tag, terminal)

    def align_nodes(self, amr):
        preterminals = {}
        alignments = amr.alignments()
        tokens = amr.tokens()
        lower = list(map(str.lower, tokens))
        for triple, node in self.nodes.items():
            indices = []
            align = alignments.get(triple)
            if align is not None:
                indices += list(map(int, align.lstrip(ALIGNMENT_PREFIX).split(ALIGNMENT_SEP)))  # split numeric
            dep = triple[2]
            if not isinstance(dep, amr_lib.Var):
                indices = self._expand_alignments(str(dep), indices, lower)
            for i in indices:
                preterminals.setdefault(i, []).append(node)
        return preterminals

    @staticmethod
    def _expand_alignments(label, orig_indices, tokens):
        # correct missing alignment by expanding to neighboring terminals contained in label
        indices = sorted(orig_indices)
        stripped = AmrConverter.strip(label, strip_sense=True, strip_quotes=True).lower()
        if indices:
            for start, offset in ((indices[0], -1), (indices[-1], 1)):  # try adding tokens around existing
                i = start + offset
                while 0 <= i < len(tokens):
                    if AmrConverter._contains_substring(stripped, tokens, indices + [i]):
                        indices.append(i)
                    elif not SKIP_TOKEN_PATTERN.match(tokens[i]):  # skip meaningless tokens
                        break
                    i += offset
            full_range = range(min(indices), max(indices) + 1)  # make this a contiguous range if valid
            if AmrConverter._contains_substring(stripped, tokens, full_range):
                indices = list(full_range)
        elif len(stripped) > 1:  # no given alignment, and label has more than one character (to avoid aligning "-")
            for i, token in enumerate(tokens):  # use any equal span, or any equal token if it occurs only once
                if stripped.startswith(token):
                    l = [i]
                    j = i
                    while j < len(tokens) - 1:
                        j += 1
                        if not SKIP_TOKEN_PATTERN.match(tokens[j]):
                            if not AmrConverter._contains_substring(stripped, tokens, l + [j]):
                                break
                            l.append(j)
                    if len(l) > 1 and stripped.endswith(tokens[l[-1]]) or tokens.count(token) == 1:
                        return l
        return indices

    @staticmethod
    def _contains_substring(label, tokens, indices):
        selected = [tokens[i] for i in sorted(indices)]
        return "".join(selected) in label or "-".join(selected) in label

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
    def _expand_names(l1):
        for node in list(l1.all):
            for edge in node:
                if edge.tag == NAME:
                    name = edge.child
                    label = resolve_label(name)
                    if label and label != CONCEPT + "(" + NAME + ")":
                        name.attrib[LABEL_ATTRIB] = CONCEPT + "(" + NAME + ")"
                        for l in AmrConverter.strip_quotes(label).split("_"):
                            l1.add_fnode(name, OP).attrib[LABEL_ATTRIB] = l if NUM_PATTERN.match(l) else '"%s"' % l

    def _update_labels(self, l1):
        for node in l1.all:
            label = resolve_label(node, reverse=True)
            if label:
                if "numbers" not in self.extensions and label.startswith(NUM + "("):
                    label = NUM + "(1)"  # replace all unresolved numbers with "1"
                elif WIKI not in self.extensions and any(e.tag == WIKI for e in node.incoming):
                    label = CONST + "(" + MINUS + ")"
            node.attrib[LABEL_ATTRIB] = label

    def to_format(self, passage, metadata=True, wikification=True, verbose=False):
        if verbose:
            print("Annotating passage..")
        textutil.annotate(passage)
        lines = ["# ::id " + passage.ID,
                 "# ::tok " + " ".join(t.text for t in passage.layer(layer0.LAYER_ID).all)] if metadata else []
        if wikification:
            if verbose:
                print("Wikifying passage...")
            WIKIFIER.wikify_passage(passage)
        if verbose:
            print("Expanding names...")
        self._expand_names(passage.layer(layer1.LAYER_ID))
        return lines + (penman.encode(penman.Graph(list(self._to_triples(passage)))).split("\n") or ["(y / yes)"])

    @staticmethod
    def _to_triples(passage):
        class _IdGenerator:
            def __init__(self):
                self._id = 0

            def __call__(self):
                self._id += 1
                return "v" + str(self._id)

        root = passage.layer(layer1.LAYER_ID).heads[0]
        pending = list(root)  # all the root's outgoing edges (NOT a list containing just the root)
        if not pending:  # there is nothing but the root node: add a dummy edge to stop the loop immediately
            pending = [namedtuple("Edge", ["parent", "child", "tag"])(root, None, None)]
        visited = set()  # to avoid cycles
        labels = defaultdict(_IdGenerator())  # to generate a different label for each variable
        prefixed_relation_counter = defaultdict(int)  # to add the index back to :op and :snt relations
        while pending:  # breadth-first search
            edge = pending.pop(0)
            if edge not in visited:  # skip cycles
                visited.add(edge)
                nodes = [edge.parent]  # nodes taking part in the relation being created
                if edge.child is not None and edge.tag not in TERMINAL_TAGS:  # skip terminals
                    nodes.append(edge.child)
                    pending += edge.child  # all the child's outgoing edges
                head_dep = []  # will be pair of (parent label, child label)
                for node in nodes:
                    label = resolve_label(node)
                    if label is None:
                        raise ValueError("Missing node label for node %s in passage %s" % (node.ID, passage.ID))
                    elif is_concept(label):  # collapsed variable + concept: create both AMR nodes and the instance-of
                        concept = None if node.ID in labels else AmrConverter.strip(label)
                        label = labels[node.ID]  # generate variable label
                        if concept is not None:  # first time we encounter the variable
                            yield label, INSTANCE, concept  # add instance-of edge
                    else:  # constant
                        label = AmrConverter.strip(label)
                    head_dep.append(label)
                if len(head_dep) > 1:
                    rel = edge.tag
                    if rel in PREFIXED_RELATION_ENUM:
                        key = (rel, edge.parent.ID)
                        prefixed_relation_counter[key] += 1
                        rel += str(prefixed_relation_counter[key])
                    elif rel == PREFIXED_RELATION_PREP:
                        rel = "-".join([rel] + list(OrderedDict.fromkeys(t.text for t in edge.child.get_terminals())))
                    yield head_dep[0], rel, head_dep[1]

    @staticmethod
    def strip(label, strip_sense=False, strip_quotes=False):  # remove type name
        label = re.sub("\w+\((.*)\)", r"\1", label)
        if strip_sense:
            label = re.sub("-\d\d$", "", label)
        if strip_quotes:
            label = AmrConverter.strip_quotes(label)
        return label

    @staticmethod
    def strip_quotes(label):
        return label[1:-1] if len(label) > 1 and label.startswith('"') and label.endswith('"') else label


def from_amr(lines, passage_id=None, return_original=False, *args, **kwargs):
    """Converts from parsed text in AMR PENMAN format to a Passage object.

    :param lines: iterable of lines in AMR PENMAN format, describing a single passage.
    :param passage_id: ID to set for passage, overriding the ID from the file
    :param return_original: return triple of (UCCA passage, AMR string, AMR ID)

    :return generator of Passage objects
    """
    del args, kwargs
    return AmrConverter().from_format(lines, passage_id, return_original=return_original)


def to_amr(passage, metadata=True, wikification=True, verbose=False, *args, **kwargs):
    """ Convert from a Passage object to a string in AMR PENMAN format (export)

    :param passage: the Passage object to convert
    :param metadata: whether to print ::id and ::tok lines
    :param wikification: whether to wikify named concepts, adding a :wiki triple
    :param verbose: whether to print extra information

    :return list of lines representing an AMR in PENMAN format, constructed from the passage
    """
    del args, kwargs
    return AmrConverter().to_format(passage, metadata, wikification, verbose)
