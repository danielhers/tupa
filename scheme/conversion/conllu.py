from ucca import convert

from .dep import DependencyConverter


class ConlluConverter(DependencyConverter, convert.ConllConverter):
    def __init__(self, *args, constituency=True, **kwargs):
        super().__init__(*args, constituency=constituency, **kwargs)

    def modify_passage(self, passage):
        passage.extra["format"] = "conllu"

    def read_line(self, line, previous_node):
        self.lines_read.append(line)
        try:
            return super().read_line(line, previous_node)
        except ValueError as e:
            raise ValueError("Failed reading line:\n" + line) from e

    def create_non_terminals(self, dep_nodes, l1):
        if not self.constituency:
            DependencyConverter.create_non_terminals(self, dep_nodes, l1)
        for dep_node in self._topological_sort(dep_nodes):
            primary_edge, *remote_edges = dep_node.incoming
            dep_node.node = dep_node.preterminal = l1.add_fnode(primary_edge.head.node, primary_edge.rel)
            if dep_node.outgoing:
                dep_node.preterminal = l1.add_fnode(dep_node.preterminal, "head")
            for edge in remote_edges:
                l1.add_remote(edge.head.node, edge.rel, dep_node.node)

    def break_cycles(self, dep_nodes):
        super().break_cycles(dep_nodes)
        for dep_node in dep_nodes:
            if not dep_node.incoming:
                dep_node.incoming = [(self.Edge(head_index=-1, rel="root", remote=False))]


def from_conllu(lines, passage_id, split=True, return_original=False, *args, **kwargs):
    """Converts from parsed text in Universal Dependencies format to a Passage object.

    :param lines: iterable of lines in Universal Dependencies format, describing a single passage.
    :param passage_id: ID to set for passage
    :param split: split each sentence to its own passage?
    :param return_original: return triple of (UCCA passage, Universal Dependencies string, sentence ID)

    :return generator of Passage objects
    """
    del args, kwargs
    return ConlluConverter().from_format(lines, passage_id, split, return_original=return_original)


def to_conllu(passage, test=False, tree=False, constituency=False, *args, **kwargs):
    """ Convert from a Passage object to a string in Universal Dependencies format (conllu)

    :param passage: the Passage object to convert
    :param test: whether to omit the head and deprel columns. Defaults to False
    :param tree: whether to omit columns for non-primary parents. Defaults to True
    :param constituency: use UCCA conversion that introduces intermediate non-terminals

    :return list of lines representing the semantic dependencies in the passage
    """
    del args, kwargs
    return ConlluConverter(constituency=constituency).to_format(passage, test, tree)
