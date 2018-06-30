from collections import deque

from operator import attrgetter
from semstr.util.amr import LABEL_ATTRIB, UNKNOWN_LABEL, LABEL_SEPARATOR
from ucca import core, layer0
from ucca.layer1 import EdgeTags

from ..config import Config


class Node:
    """
    Temporary representation for core.Node with only relevant information for parsing
    """
    def __init__(self, index, swap_index=None, orig_node=None, text=None, paragraph=None, tag=None, label=None,
                 implicit=False, is_root=False, root=None):
        self.index = index  # Index in the configuration's node list
        self.orig_node = orig_node  # Associated core.Node from the original Passage, during training
        self.node_id = orig_node.ID if orig_node else None  # ID of the original node
        self.text = text  # Text for terminals, None for non-terminals
        self.paragraph = paragraph  # int for terminals, None for non-terminals
        self.tag = tag  # Node tag of the original node (Word/Punctuation)
        if label is None:
            self.label = self.category = None
        else:  # Node label prediction is enabled
            self.label, _, self.category = label.partition(LABEL_SEPARATOR)
            if not self.category:
                self.category = None
        # Whether a label has been set yet (necessary because None is a valid label too):
        self.labeled = self.orig_node is not None and self.orig_node.attrib.get(LABEL_ATTRIB) is None
        self.node_index = int(self.node_id.split(core.Node.ID_SEPARATOR)[1]) if orig_node else None
        self.outgoing = []  # Edge list
        self.incoming = []  # Edge list
        self.children = []  # Node list: the children of all edges in outgoing
        self.parents = []  # Node list: the parents of all edges in incoming
        self.outgoing_tags = set()  # String set
        self.incoming_tags = set()  # String set
        self.node = None  # Associated core.Node, when creating final Passage
        self.implicit = implicit  # True or False
        self.swap_index = self.index if swap_index is None else swap_index  # To avoid swapping nodes more than once
        self.height = 0
        self._terminals = None
        self.is_root = is_root
        self.root = root  # Original Passage object this belongs to

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
    def attach_nodes(l0, l1, nodes, labeled=True, node_labels=False, verify=False):
        remotes = []  # To be handled after all nodes are created
        linkages = []  # To be handled after all non-linkage nodes are created
        for node in Node.topological_sort(nodes):
            if labeled and verify:
                assert node.text or node.outgoing or node.implicit, "Non-terminal leaf node: %s" % node
            if node.is_linkage:
                linkages.append(node)
            else:
                for edge in node.outgoing:
                    if edge.remote:
                        remotes.append((node, edge))
                    else:
                        edge.child.add_to_l1(l0, l1, node, edge.tag, labeled, node_labels)
        Node.attach_remotes(l1, remotes, verify)
        Node.attach_linkages(l1, linkages, verify)

    @staticmethod
    def topological_sort(nodes):
        """
        Sort self.nodes topologically, each node appearing as early as possible
        Also sort each node's outgoing and incoming edge according to the node order
        """
        levels = {}
        level_by_index = {}
        stack = [node for node in nodes if not node.outgoing]
        while stack:
            node = stack.pop()
            if node.index not in level_by_index:
                parents = [edge.parent for edge in node.incoming]
                if parents:
                    unexplored_parents = [parent for parent in parents
                                          if parent.index not in level_by_index]
                    if unexplored_parents:
                        for parent in unexplored_parents:
                            stack.append(node)
                            stack.append(parent)
                    else:
                        level = 1 + max(level_by_index[parent.index] for parent in parents)
                        levels.setdefault(level, []).append(node)
                        level_by_index[node.index] = level
                else:
                    levels.setdefault(0, []).append(node)
                    level_by_index[node.index] = 0
        nodes = [node for level, level_nodes in sorted(levels.items())
                 for node in sorted(level_nodes, key=lambda x: x.node_index or x.index)]
        for node in nodes:
            node.outgoing.sort(key=lambda x: x.child.node_index or nodes.index(x.child))
            node.incoming.sort(key=lambda x: x.parent.node_index or nodes.index(x.parent))
        return nodes

    def add_to_l1(self, l0, l1, parent, tag, labeled, node_labels):
        """
        Called when creating final Passage to add a new core.Node
        :param l0: Layer0 of the passage
        :param l1: Layer1 of the passage
        :param parent: node
        :param tag: edge tag to link to parent
        :param labeled: there is a reference passage, so keep original node IDs in the "remarks" field
        :param node_labels: whether to add a node label
        """
        edge = self.outgoing[0] if len(self.outgoing) == 1 else None
        if self.text:  # For Word terminals (Punctuation already created by add_punct for parent)
            if parent.node is not None:
                if self.node is None:
                    self.node = parent.node.add(EdgeTags.Terminal, self.get_terminal(l0)).child
                elif self.node not in parent.node.children:
                    parent.node.add(EdgeTags.Terminal, self.node)
        elif edge and edge.child.text and layer0.is_punct(edge.child.get_terminal(l0)):
            if Config().args.verify:
                assert tag == EdgeTags.Punctuation, "Punctuation parent %s's edge tag is %s" % (parent.node_id, tag)
                assert edge.tag == EdgeTags.Terminal, "Punctuation %s's edge tag is %s" % (self.node_id, edge.tag)
            if self.node is None:
                self.node = l1.add_punct(parent.node, edge.child.get_terminal(l0))
                edge.child.node = self.node[0].child
            elif parent.node is not None and self.node not in parent.node.children:
                parent.node.add(EdgeTags.Punctuation, self.node)
        else:  # The usual case
            assert self.node is None, "Trying to create the same node twice (multiple incoming primary edges): " + \
                                      ", ".join(map(str, self.incoming))
            if parent is not None and parent.label and parent.node is None:  # If parent is an orphan and has a a label,
                parent.add_to_l1(l0, l1, None, Config().args.orphan_label, labeled, node_labels)  # link to root
            self.node = l1.add_fnode(None if parent is None else parent.node, tag, implicit=self.implicit)
        if labeled:  # In training
            self.set_node_id()
        if node_labels:
            self.set_node_label()

    @staticmethod
    def attach_remotes(l1, remotes, verify=False):
        for node, edge in remotes:  # Add remote edges
            try:
                assert node.node is not None, "Remote edge from nonexistent node"
                assert edge.child.node is not None, "Remote edge to nonexistent node"
                l1.add_remote(node.node, edge.tag, edge.child.node)
            except AssertionError:
                if verify:
                    raise

    @staticmethod
    def attach_linkages(l1, linkages, verify=False):
        for node in linkages:  # Add linkage nodes and edges
            try:
                link_relation = None
                link_args = []
                for edge in node.outgoing:
                    assert edge.child.node, "Linkage edge to nonexistent node"
                    if edge.tag == EdgeTags.LinkRelation:
                        assert not link_relation, "Multiple link relations: %s, %s" % (link_relation, edge.child.node)
                        link_relation = edge.child.node
                    elif edge.tag == EdgeTags.LinkArgument:
                        link_args.append(edge.child.node)
                    else:
                        Config().log("Ignored non-linkage edge %s from linkage node %s" % (edge, node))
                assert link_relation is not None, "No link relations: %s" % node
                # if len(link_args) < 2:
                #     Config().log("Less than two link arguments for linkage node %s" % node)
                node.node = l1.add_linkage(link_relation, *link_args)
                if node.node_id:  # We are in training and we have a gold passage
                    node.node.extra["remarks"] = node.node_id  # For reference
            except AssertionError:
                if verify:
                    raise

    def get_terminal(self, l0):
        return l0.by_position(self.index)

    def set_node_id(self):
        if self.node is not None and self.node_id is not None:
            self.node.extra["remarks"] = self.node_id  # Keep original node ID for reference

    def set_node_label(self):
        if self.node is not None:
            self.node.attrib[LABEL_ATTRIB] = self.label or UNKNOWN_LABEL

    @property
    def is_linkage(self):
        """
        Is this a LKG type node? (During parsing there are no node types)
        """
        return self.outgoing_tags and self.outgoing_tags.intersection((EdgeTags.LinkRelation, EdgeTags.LinkArgument))

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

    @property
    def tok(self):
        return self.orig_node.tok

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
        return self.index == other.index and self.outgoing == other.outgoing

    def __hash__(self):
        return hash((self.index, tuple(self.outgoing)))

    def __iter__(self):
        return iter(self.outgoing)
