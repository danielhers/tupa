from ucca import convert


class DependencyConverter(convert.DependencyConverter):
    """
    Alternative converter to the one in UCCA - instead of introducing centers etc. to get a proper constituency
    structure, just copy the exact structure from the dependency graph, with all edges being between terminals (+root)
    """
    TOP = "TOP"

    def __init__(self, *args, constituency=False, **kwargs):
        super(DependencyConverter, self).__init__(*args, **kwargs)
        self.constituency = constituency
        self.lines_read = []

    def create_non_terminals(self, dep_nodes, l1):
        if self.constituency:
            super(DependencyConverter, self).create_non_terminals(dep_nodes, l1)
        for dep_node in dep_nodes:
            if dep_node.position != 0 and not dep_node.incoming and dep_node.outgoing:
                dep_node.node = dep_node.preterminal = l1.add_fnode(None, self.TOP if dep_node.is_top else self.ROOT)
        for dep_node in self._topological_sort(dep_nodes):
            # create pre-terminals and edges
            incoming = list(dep_node.incoming)
            if dep_node.is_top and incoming[0].head_index != 0:
                top_edge = self.Edge(0, self.TOP, False)
                top_edge.head = dep_nodes[0]
                incoming[:0] = [top_edge]
            edge = incoming[0]
            dep_node.node = dep_node.preterminal = l1.add_fnode(edge.head.node, edge.rel)
            for edge in incoming[1:]:
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
        return [e for e in unit.incoming if e.tag not in (self.ROOT, self.TOP)]

    def break_cycles(self, dep_nodes):
        for dep_node in dep_nodes:
            for edge in dep_node.incoming:
                edge.remote = False

    def is_top(self, unit):
        if not unit.outgoing and unit.incoming:  # go to pre-terminal
            unit = unit.parents[0]
        return any(e.tag == self.TOP for e in unit.incoming)
