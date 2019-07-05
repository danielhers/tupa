from collections import deque
from operator import attrgetter

from semstr.util.amr import UNKNOWN_LABEL, LABEL_SEPARATOR


class Node:
    """
    Temporary representation for core.Node with only relevant information for parsing
    """
    def __init__(self, index, swap_index=None, orig_node=None, text=None, label=None,
                 implicit=False, is_root=False, root=None):
        self.index = index  # Index in the configuration's node list
        self.orig_node = orig_node  # Associated core.Node from the original Graph, during training
        self.id = str(orig_node.id) if orig_node else index  # ID of the original node
        self.text = text  # Text for terminals, None for non-terminals
        if label is None:
            self.label = self.category = None
        else:  # Node label prediction is enabled
            self.label, _, self.category = label.partition(LABEL_SEPARATOR)
            if not self.category:
                self.category = None
        # Whether a label has been set yet (necessary because None is a valid label too):
        self.labeled = self.orig_node is not None and self.orig_node.label is None
        self.node_index = int(self.id) if orig_node else None
        self.outgoing = []  # Edge list
        self.incoming = []  # Edge list
        self.children = []  # Node list: the children of all edges in outgoing
        self.parents = []  # Node list: the parents of all edges in incoming
        self.outgoing_tags = set()  # String set
        self.incoming_tags = set()  # String set
        self.node = None  # Associated core.Node, when creating final Graph
        self.implicit = implicit  # True or False
        self.swap_index = self.index if swap_index is None else swap_index  # To avoid swapping nodes more than once
        self.height = 0
        self._terminals = None
        self.is_root = is_root
        self.root = root  # Original Graph object this belongs to

    def get(self, prop):
        for p, v in zip(self.orig_node.properties, self.orig_node.values):
            if p == prop:
                return v

    def add_incoming(self, edge):
        self.incoming.append(edge)
        self.parents.append(edge.parent)
        self.incoming_tags.add(edge.tag)

    def add_outgoing(self, edge):
        self.outgoing.append(edge)
        self.children.append(edge.child)
        self.outgoing_tags.add(edge.tag)
        self.height = max(self.height, edge.child.height + 1)
        self._terminals = None  # Invalidate terminals because we might have added some

    @staticmethod
    def attach_nodes(graph, nodes):
        for node in nodes:
            node.node = graph.add_node(node.id)
            node.set_node_label()
        for node in nodes:
            for edge in node.outgoing:
                graph.add_edge(edge.parent.id, edge.child.id, edge.tag)

    def set_node_label(self):
        if self.node is not None:
            self.node.label = self.label or UNKNOWN_LABEL

    @property
    def descendants(self):
        """
        Find all children of this node recursively
        """
        result = [self]
        queue = deque(node for node in self.children if node is not self)
        while queue:
            node = queue.popleft()
            if node is not self and node not in result:
                queue.extend(node.children)
                result.append(node)
        return result

    @property
    def terminals(self):
        if self._terminals is None:
            q = [self]
            terminals = []
            while q:
                n = q.pop()
                q.extend(n.children)
                if n.text is not None:
                    terminals.append(n)
            self._terminals = sorted(terminals, key=attrgetter("index"))
        return self._terminals

    def __repr__(self):
        return Node.__name__ + "(" + str(self.index) + \
               ((", " + self.text) if self.text else "") + \
               ((", " + self.id) if self.id else "") + ")"

    def __str__(self):
        s = '"%s"' % self.text if self.text else str(self.id) or str(self.index)
        if self.label:
            s += "/" + self.label
        return s

    def __eq__(self, other):
        return self.index == other.index and self.outgoing == other.outgoing

    def __hash__(self):
        return hash((self.index, tuple(self.outgoing)))

    def __iter__(self):
        return iter(self.outgoing)
