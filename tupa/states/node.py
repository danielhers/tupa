from collections import deque
from operator import attrgetter

from tupa.config import Config
from ucca import core, layer0
from ucca.layer1 import EdgeTags


class Node(object):
    """
    Temporary representation for core.Node with only relevant information for parsing
    """
    def __init__(self, index, swap_index=None, orig_node=None, text=None, paragraph=None, tag=None, label=None,
                 implicit=False, pos_tag=None, dep_rel=None, dep_head=None):
        self.index = index  # Index in the configuration's node list
        self.orig_node = orig_node  # Associated core.Node from the original Passage, during training
        self.node_id = orig_node.ID if orig_node else None  # ID of the original node
        self.text = text  # Text for terminals, None for non-terminals
        self.paragraph = paragraph  # int for terminals, None for non-terminals
        self.tag = tag  # Node tag of the original node (Word/Punctuation)
        self.label = label  # Node label (if node label prediction is enabled)
        self.node_index = int(self.node_id.split(core.Node.ID_SEPARATOR)[1]) if orig_node else None
        self.outgoing = []  # Edge list
        self.incoming = []  # Edge list
        self.children = []  # Node list: the children of all edges in outgoing
        self.parents = []  # Node list: the parents of all edges in incoming
        self.outgoing_tags = set()  # String set
        self.incoming_tags = set()  # String set
        self.node = None  # Associated core.Node, when creating final Passage
        self.implicit = implicit  # True or False
        self.pos_tag = pos_tag
        self.dep_rel = dep_rel
        self.dep_head = dep_head
        self.swap_index = self.index if swap_index is None else swap_index  # To avoid swapping nodes more than once
        self.height = 0
        self._terminals = None

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

    def add_to_l1(self, l1, parent, tag, terminals, labeled):
        """
        Called when creating final Passage to add a new core.Node
        :param l1: Layer1 of the passage
        :param parent: node
        :param tag: edge tag to link to parent
        :param terminals: all terminals strings in the passage
        :param labeled: there is a reference passage, so keep original node IDs in the "remarks" field
        """
        if Config().args.verify:
            assert self.node is None or self.text is not None,\
                "Trying to create the same node twice: %s, parent: %s" % (self.node.ID, parent)
        edge = self.outgoing[0] if len(self.outgoing) == 1 else None
        if self.text:  # For Word terminals (Punctuation already created by add_punct for parent)
            if self.node is None and parent.node is not None:
                self.node = parent.node.add(EdgeTags.Terminal,
                                            terminals[self.index - 1]).child
        elif edge and edge.child.text and layer0.is_punct(terminals[edge.child.index - 1]):
            if Config().args.verify:
                assert tag == EdgeTags.Punctuation, "Tag for %s is %s" % (parent.node_id, tag)
                assert edge.tag == EdgeTags.Terminal, "Tag for %s is %s" % (self.node_id, edge.tag)
            self.node = l1.add_punct(parent.node, terminals[edge.child.index - 1])
            edge.child.node = self.node[0].child
        else:  # The usual case
            if parent.node is None:  # If parent is an orphan, attach it to the root with F label
                parent.node = l1.add_fnode(None, EdgeTags.Function)
            self.node = l1.add_fnode(parent.node, tag, implicit=self.implicit)
        if labeled and self.node is not None and self.node_id is not None:  # In training
            self.node.extra["remarks"] = self.node_id  # Keep original node ID for reference
        if self.label is not None:
            self.node.attrib[Config().args.node_label_attrib] = self.label

    @property
    def is_linkage(self):
        """
        Is this a LKG type node? (During parsing there are no node types)
        """
        return self.outgoing_tags and self.outgoing_tags.issubset((EdgeTags.LinkRelation, EdgeTags.LinkArgument))

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
               ((", " + self.node_id) if self.node_id else "") + ")"

    def __str__(self):
        s = '"%s"' % self.text if self.text else self.node_id or str(self.index)
        if self.label:
            s += "/" + self.label
        return s

    def __eq__(self, other):
        return self.index == other.index and self.outgoing == other.outgoing or \
                self.label is not None and self.label == other.label

    def __hash__(self):
        return hash(self.label if self.label is not None else (self.index, tuple(self.outgoing)))
