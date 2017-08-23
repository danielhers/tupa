from ucca import convert


class DependencyConverter(convert.DependencyConverter):
    """
    Alternative converter to the one in UCCA - instead of introducing centers etc. to get a proper constituency
    structure, just copy the exact structure from the dependency graph, with all edges being between terminals (+root)
    """
    def __init__(self, *args, constituency=False, **kwargs):
        super(DependencyConverter, self).__init__(*args, **kwargs)
        self.constituency = constituency
        self.lines_read = []

    def create_non_terminals(self, dep_nodes, l1):
        if self.constituency:
            return super(DependencyConverter, self).create_non_terminals(dep_nodes, l1)
        for dep_node in dep_nodes:
            if dep_node.incoming:
                # create pre-terminals and edges
                edge = dep_node.incoming[0]
                dep_node.node = dep_node.preterminal = l1.add_fnode(edge.head.node, edge.rel)
                for edge in dep_node.incoming[1:]:
                    l1.add_remote(edge.head.node, edge.rel, dep_node.node)

    def from_format(self, lines, passage_id, split=False, return_original=False):
        for passage in super(DependencyConverter, self).from_format(lines, passage_id, split=split):
            yield (passage, self.lines_read, passage.ID) if return_original else passage
            self.lines_read = []

    def find_head_terminal(self, unit):
        while unit.outgoing:  # still non-terminal
            unit = unit.children[0]
        return unit

    def find_top_headed_edges(self, unit):
        if not unit.outgoing and unit.incoming:  # go to pre-terminal
            unit = unit.parents[0]
        return unit.incoming

    def break_cycles(self, dep_nodes):
        for dep_node in dep_nodes:
            for edge in dep_node.incoming:
                edge.remote = False
