from ucca import convert


class DependencyConverter(convert.DependencyConverter):
    """
    Alternative converter to the one in UCCA - instead of introducing centers etc. to get a proper constituency
    structure, just copy the exact structure from the dependency graph, with all edges being between terminals (+root)
    """
    TOP = "TOP"
    HEAD = "head"

    def __init__(self, *args, constituency=False, tree=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.constituency = constituency
        self.tree = tree
        self.lines_read = []

    def read_line_and_append(self, read_line, line, previous_node):
        self.lines_read.append(line)
        try:
            return read_line(line, previous_node)
        except ValueError as e:
            raise ValueError("Failed reading line:\n" + line) from e

    def split_line(self, line):
        return line.split("\t")

    def create_non_terminals(self, dep_nodes, l1):
        if self.constituency:
            super().create_non_terminals(dep_nodes, l1)
        if not self.tree:
            for dep_node in dep_nodes:  # Create top nodes
                if dep_node.position != 0 and not dep_node.incoming and dep_node.outgoing:
                    dep_node.node = dep_node.preterminal = l1.add_fnode(None, (self.ROOT, self.TOP)[dep_node.is_top])
        for dep_node in self._topological_sort(dep_nodes):  # Create all other nodes
            incoming = list(dep_node.incoming)
            if dep_node.is_top and incoming[0].head_index != 0:
                top_edge = self.Edge(head_index=0, rel=self.TOP, remote=False)
                top_edge.head = dep_nodes[0]
                incoming[:0] = [top_edge]
            primary_edge, *remote_edges = incoming
            dep_node.node = dep_node.preterminal = None if primary_edge.rel.upper() == self.ROOT else \
                l1.add_fnode(primary_edge.head.node, primary_edge.rel)
            if dep_node.outgoing:
                dep_node.preterminal = l1.add_fnode(dep_node.preterminal, self.HEAD)
            for edge in remote_edges:
                if primary_edge.head.node != edge.head.node:  # Avoid multi-edges
                    l1.add_remote(edge.head.node or l1.heads[0], edge.rel, dep_node.node)

    def from_format(self, lines, passage_id, split=False, return_original=False):
        for passage in super().from_format(lines, passage_id, split=split):
            yield (passage, self.lines_read, passage.ID) if return_original else passage
            self.lines_read = []

    def find_head_terminal(self, unit):
        while unit.outgoing:  # still non-terminal
            heads = [e.child for e in unit.outgoing if e.tag == self.HEAD]
            try:
                unit = heads[0] if heads else next(iter(e.child for e in unit.outgoing if not e.attrib.get("remote") and
                                                        not e.child.attrib.get("implicit")))
            except StopIteration:
                unit = unit.children[0]
        return unit

    def find_top_headed_edges(self, unit):
        return [e for e in self.find_headed_unit(unit).incoming if e.tag not in (self.ROOT, self.TOP)]

    def break_cycles(self, dep_nodes):
        for dep_node in dep_nodes:
            if dep_node.incoming:
                for edge in dep_node.incoming:
                    edge.remote = False
            elif self.tree:
                dep_node.incoming = [(self.Edge(head_index=-1, rel=self.ROOT.lower(), remote=False))]

    def is_top(self, unit):
        return any(e.tag == self.TOP for e in self.find_headed_unit(unit).incoming)

    def find_headed_unit(self, unit):
        while unit.incoming and (not unit.outgoing or unit.incoming[0].tag == self.HEAD):
            unit = unit.parents[0]
        return unit
