import sys
from collections import deque, defaultdict

from ucca import core, layer0, layer1, textutil
from ucca.layer1 import EdgeTags

from scheme.constrain import CONSTRAINTS
from scheme.constraints import Constraints
from scheme.constraints import Direction
from scheme.util.amr import LABEL_ATTRIB
from .edge import Edge
from .node import Node
from ..action import Actions
from ..config import Config


class InvalidActionError(AssertionError):
    def __init__(self, *args, is_type=False):
        super().__init__(*args)
        self.is_type = is_type


class State(object):
    """
    The parser's state, responsible for applying actions and creating the final Passage
    :param passage: a Passage object to get the tokens from, and everything else if training
    """
    def __init__(self, passage):
        self.args = Config().args
        self.constraints = CONSTRAINTS.get(passage.extra.get("format"), Constraints)(self.args)
        self.log = []
        self.finished = False
        self.passage = passage
        l0 = passage.layer(layer0.LAYER_ID)
        l1 = passage.layer(layer1.LAYER_ID)
        self.labeled = any(n.outgoing or n.attrib.get(LABEL_ATTRIB) for n in l1.all)
        self.terminals = [Node(i, orig_node=t, root=passage, text=t.text, paragraph=t.paragraph, tag=t.tag,
                               pos_tag=t.extra.get(textutil.TAG_KEY),
                               dep_rel=t.extra.get(textutil.DEP_KEY),
                               dep_head=t.extra.get(textutil.HEAD_KEY),
                               ner_type=t.extra.get(textutil.NER_KEY),
                               ner_iob=t.extra.get(textutil.IOB_KEY),
                               lemma=t.extra.get(textutil.LEMMA_KEY))
                          for i, t in enumerate(l0.all, start=1)]
        self.stack = []
        self.buffer = deque()
        self.nodes = []
        self.heads = set()
        self.root = self.add_node(orig_node=l1.heads[0], is_root=True)  # Root is not in the buffer
        self.stack.append(self.root)
        self.buffer += self.terminals
        self.nodes += self.terminals
        self.actions = []  # History of applied actions
        self.type_validity_cache = {}
        self.need_label = False  # Whether we are waiting for label_node() to be called

    def is_valid_action(self, action):
        """
        :param action: action to check for validity
        :return: is the action (including tag) valid in the current state?
        """
        valid = self.type_validity_cache.get(action.type_id)
        if valid is None:
            try:
                self.check_valid_action(action)
                valid = True
            except InvalidActionError as e:
                valid = False
                if e.is_type:
                    self.type_validity_cache[action.type_id] = valid
        return valid

    def check_valid_action(self, action, message=False):
        """
        Raise InvalidActionError if the action is invalid in the current state
        :param action: action to check for validity
        :param message: whether to add an informative message to the thrown exception
        """
        def _check_possible_node():
            self.check(self.node_ratio() < self.args.max_node_ratio,
                       message and "Non-terminals/terminals ratio: %.3f" % self.args.max_node_ratio, is_type=True)
            for head in self.heads:
                self.check(head.height <= self.args.max_height,
                           message and "Graph height: %d" % self.args.max_height, is_type=True)

        def _check_possible_parent(node):
            self.check(node.text is None, message and "Terminals may not have children: %s" % node.text, is_type=True)
            if self.args.constraints:
                for rule in self.constraints.tag_rules:
                    violation = rule.violation(node, action.tag, Direction.outgoing, message=message)
                    self.check(violation is None, violation)
                self.check(self.constraints.allow_parent(node, action.tag),
                           message and "%s may not be a '%s' parent (currently %s)" % (
                               node, action.tag, ", ".join(map(str, node.outgoing)) or "childless"))
                self.check(not self.constraints.require_implicit_childless or not node.implicit,
                           message and "Implicit nodes may not have children: %s" % s0, is_type=True)

        def _check_possible_child(node):
            self.check(node is not self.root, message and "Root may not have parents", is_type=True)
            if self.args.constraints:
                self.check((node.text is not None) == (action.tag == EdgeTags.Terminal),
                           message and "Edge tag must be %s iff child is terminal, but node is %s and edge tag is %s" %
                           (EdgeTags.Terminal, node, action.tag))
                for rule in self.constraints.tag_rules:
                    violation = rule.violation(node, action.tag, Direction.incoming, message=message)
                    self.check(violation is None, violation)
                self.check(self.constraints.allow_child(node, action.tag),
                           message and "%s may not be a '%s' child (currently %s, %s)" % (
                               node, action.tag, ", ".join(map(str, node.incoming)) or "parentless",
                               ", ".join(map(str, node.outgoing)) or "childless"))
                self.check(self.constraints.possible_multiple_incoming is None or
                           action.remote or action.tag in self.constraints.possible_multiple_incoming or
                           all(e.remote or e.tag in self.constraints.possible_multiple_incoming for e in node.incoming),
                           message and "Multiple non-remote '%s' parents not allowed for %s" % (action.tag, node))

        def _check_possible_edge():
            parent, child = self.get_parent_child(action)
            _check_possible_parent(parent)
            _check_possible_child(child)
            if self.args.constraints:
                if parent is self.root:
                    self.check(self.constraints.allow_root_terminal_children or child.text is None,
                               message and "Terminal child '%s' for root" % child, is_type=True)
                    self.check(self.constraints.top_level_allowed is None or
                               action.tag in self.constraints.top_level_allowed,
                               message and "Root may not have %s edges" % action.tag)
                else:
                    self.check(self.constraints.top_level_only is None or
                               action.tag not in self.constraints.top_level_only,
                               message and "Only root may have %s edges" % action.tag)
                edge = Edge(parent, child, action.tag, remote=action.remote)
                self.check(self.constraints.allow_edge(edge), message and "Edge not allowed: %s" % edge)
            self.check(parent not in child.descendants,
                       message and "Detected cycle by edge: %s->%s" % (parent, child), is_type=True)

        if self.args.constraints:
            self.check(self.constraints.allow_action(action, self.actions),
                       message and "Action not allowed: %s " % action + (
                           ("after " + ", ".join("%s" % a for a in self.actions[-3:])) if self.actions else "as first"))
        if action.is_type(Actions.Finish):
            self.check(not self.buffer, "May only finish at the end of the input buffer", is_type=True)
            if self.args.swap:  # Without swap, the oracle may be incapable even of single action
                self.check(self.root.outgoing or all(n is self.root or n.is_linkage or n.text for n in self.nodes),
                           message and "Root has no child at parse end", is_type=True)
            for n in self.nodes:
                self.check(not self.args.require_connected or n is self.root or n.is_linkage or n.text or
                           n.incoming, message and "Non-terminal %s has no parent at parse end" % n, is_type=True)
                self.check(not self.args.node_labels or n.text or n.labeled,
                           message and "Non-terminal %s has no label at parse end" % n, is_type=True)
        else:
            self.check(self.action_ratio() < self.args.max_action_ratio,
                       message and "Actions/terminals ratio: %.3f" % self.args.max_action_ratio, is_type=True)
            if action.is_type(Actions.Shift):
                self.check(self.buffer, message and "Shifting from empty buffer", is_type=True)
            else:  # Unary actions
                self.check(self.stack, message and "Empty stack at %s" % action, is_type=True)
                s0 = self.stack[-1]
                if action.is_type(Actions.Node, Actions.RemoteNode):
                    _check_possible_child(s0)
                    _check_possible_node()
                elif action.is_type(Actions.Implicit):
                    _check_possible_parent(s0)
                    _check_possible_node()
                elif action.is_type(Actions.Label):
                    self.check(self.args.node_labels, message and "Node labels disabled", is_type=True)
                    try:
                        node = self.stack[-1 - (action.tag or 0)]
                    except IndexError:
                        node = None
                    self.check(node is not None, message and "Labeling invalid node %s when stack size is %d" % (
                        action.tag, len(self.stack)))
                    self.check(not node.labeled, message and "Labeling already-labeled node: %s" % node, is_type=True)
                    self.check(node.text is None, message and "Terminals do not have labels: %s" % node, is_type=True)
                elif action.is_type(Actions.Reduce):
                    if s0 is self.root:
                        self.check(self.root.labeled or not self.args.node_labels,
                                   message and "Reducing root without label", is_type=True)
                    else:
                        self.check(not self.args.require_connected or s0.is_linkage or s0.text or s0.incoming,
                                   message and "Reducing parentless non-terminal %s" % s0, is_type=True)
                    self.check(not self.args.node_labels or s0.text or s0.labeled,
                               message and "Reducing non-terminal %s without label" % s0, is_type=True)
                else:  # Binary actions
                    self.check(len(self.stack) > 1, message and "%s with len(stack) < 2" % action, is_type=True)
                    if action.is_type(Actions.LeftEdge, Actions.RightEdge, Actions.LeftRemote, Actions.RightRemote):
                        _check_possible_edge()
                    elif action.is_type(Actions.Swap):
                        # A regular swap is possible since the stack has at least two elements;
                        # A compound swap is possible if the stack is longer than the distance
                        distance = action.tag or 1
                        self.check(1 <= distance < len(self.stack), message and "Invalid swap distance: %d" % distance)
                        swapped = self.stack[-distance - 1]
                        # To prevent swap loops: only swap if the nodes are currently in their original order
                        self.check(self.swappable(s0, swapped),
                                   message and "Already swapped nodes: %s (swap index %g) <--> %s (swap index %g)"
                                   % (swapped, swapped.swap_index, s0, s0.swap_index))
                    else:
                        raise ValueError("Invalid action: %s" % action)

    @staticmethod
    def swappable(right, left):
        return left.swap_index < right.swap_index

    def is_valid_label(self, label):
        """
        :param label: label to check for validity
        :return: is the label valid in the current state?
        """
        try:
            self.check_valid_label(label)
        except InvalidActionError:
            return False
        return True

    def check_valid_label(self, label, message=False):
        if self.args.constraints and label is not None:
            valid = self.constraints.allow_label(self.stack[self.need_label], label)
            self.check(valid, message and "May not label %s as %s: %s" % (self.stack[self.need_label], label, valid))

    @staticmethod
    def check(condition, *args, **kwargs):
        if not condition:
            raise InvalidActionError(*args, **kwargs)

    # noinspection PyTypeChecker
    def transition(self, action):
        """
        Main part of the parser: apply action given by oracle or classifier
        :param action: Action object to apply
        """
        action.apply()
        self.log = []
        if action.is_type(Actions.Shift):  # Push buffer head to stack; shift buffer
            self.stack.append(self.buffer.popleft())
        elif action.is_type(Actions.Node, Actions.RemoteNode):  # Create new parent node and add to the buffer
            action.node = self.add_node(orig_node=action.orig_node)
            self.add_edge(Edge(action.node, self.stack[-1], action.tag, remote=action.remote))
            self.buffer.appendleft(action.node)
        elif action.is_type(Actions.Implicit):  # Create new child node and add to the buffer
            action.node = self.add_node(orig_node=action.orig_node, implicit=True)
            self.add_edge(Edge(self.stack[-1], action.node, action.tag))
            self.buffer.appendleft(action.node)
        elif action.is_type(Actions.Label):
            self.need_label = -1 - (action.tag or 0)  # The parser is responsible to choose a label and set it
        elif action.is_type(Actions.Reduce):  # Pop stack (no more edges to create with this node)
            self.stack.pop()
        elif action.is_type(Actions.LeftEdge, Actions.LeftRemote, Actions.RightEdge, Actions.RightRemote):
            parent, child = self.get_parent_child(action)
            action.edge = self.add_edge(Edge(parent, child, action.tag, remote=action.remote))
        elif action.is_type(Actions.Swap):  # Place second (or more) stack item back on the buffer
            distance = action.tag or 1
            s = slice(-distance - 1, -1)
            self.log.append("%s <--> %s" % (", ".join(map(str, self.stack[s])), self.stack[-1]))
            self.buffer.extendleft(reversed(self.stack[s]))  # extendleft reverses the order
            del self.stack[s]
        elif action.is_type(Actions.Finish):  # Nothing left to do
            self.finished = True
        else:
            raise ValueError("Invalid action: %s" % action)
        if self.args.verify:
            intersection = set(self.stack).intersection(self.buffer)
            assert not intersection, "Stack and buffer overlap: %s" % intersection
        action.index = len(self.actions)
        self.actions.append(action)
        self.type_validity_cache = {}

    def add_node(self, **kwargs):
        """
        Called during parsing to add a new Node (not core.Node) to the temporary representation
        :param kwargs: keyword arguments for Node()
        """
        node = Node(len(self.nodes), swap_index=self.calculate_swap_index(), root=self.passage, **kwargs)
        if self.args.verify:
            assert node not in self.nodes, "Node already exists"
        self.nodes.append(node)
        self.heads.add(node)
        self.log.append("node: %s (swap_index: %g)" % (node, node.swap_index))
        return node

    def calculate_swap_index(self):
        """
        Update a new node's swap index according to the nodes before and after it.
        Usually the swap index is just the index, i.e., len(self.nodes).
        If the buffer is not empty and its head is not a terminal, it means that it is a non-terminal created before.
        In that case, the buffer head's index will be lower than the new node's index, so the new node's swap index will
        be the arithmetic mean between the previous node (stack top) and the next node (buffer head).
        Then, in the validity check on the SWAP action, we will correctly identify this node as always having appearing
        before the current buffer head. Otherwise, we would prevent swapping them even though it should be valid
        (because they have never been swapped before).
        """
        if self.buffer:
            b = self.buffer[0]
            if self.stack and (b.text is not None or b.swap_index <= len(self.nodes)):
                s = self.stack[-1]
                return (s.swap_index + b.swap_index) / 2
        return None

    def add_edge(self, edge):
        edge.add()
        self.heads.discard(edge.child)
        self.log.append("edge: %s" % edge)
        return edge

    def get_parent_child(self, action):
        if action.is_type(Actions.LeftEdge, Actions.LeftRemote):
            return self.stack[-1], self.stack[-2]
        elif action.is_type(Actions.RightEdge, Actions.RightRemote):
            return self.stack[-2], self.stack[-1]
        else:
            return None, None

    def label_node(self, label):
        node = self.stack[self.need_label]
        node.label = label
        node.labeled = True
        self.log.append("label: %s" % node)
        self.type_validity_cache = {}
        self.need_label = False

    def create_passage(self, verify=True):
        """
        Create final passage from temporary representation
        :param verify: fail if this results in an improper passage
        :return: core.Passage created from self.nodes
        """
        if self.args.verbose > 2:
            print("Creating passage %s from state..." % self.passage.ID)
        passage = core.Passage(self.passage.ID)
        passage_format = self.passage.extra.get("format")
        if passage_format:
            passage.extra["format"] = passage_format
        l0 = layer0.Layer0(passage)
        terminals = [l0.add_terminal(text=terminal.text, punct=terminal.tag == layer0.NodeTags.Punct,
                                     paragraph=terminal.paragraph) for terminal in self.terminals]
        l1 = layer1.Layer1(passage)
        self.root.node = l1.heads[0]
        if self.args.node_labels:
            self.root.set_node_label()
        if self.labeled:  # We have a reference passage
            self.root.set_node_id()
            self.fix_terminal_tags(terminals)
        remotes = []  # To be handled after all nodes are created
        linkages = []  # To be handled after all non-linkage nodes are created
        self.topological_sort()  # Sort self.nodes
        for node in self.nodes:
            if self.labeled and verify:
                assert node.text or node.outgoing or node.implicit, "Non-terminal leaf node: %s" % node
            if node.is_linkage:
                linkages.append(node)
            else:
                for edge in node.outgoing:
                    if edge.remote:
                        remotes.append((node, edge))
                    else:
                        edge.child.add_to_l1(l1, node, edge.tag, terminals, self.labeled, self.args.node_labels)

        for node, edge in remotes:  # Add remote edges
            try:
                assert node.node is not None, "Remote edge from nonexistent node"
                assert edge.child.node is not None, "Remote edge to nonexistent node"
                l1.add_remote(node.node, edge.tag, edge.child.node)
            except AssertionError:
                if verify:
                    raise

        for node in linkages:  # Add linkage nodes and edges
            try:
                link_relation = None
                link_args = []
                for edge in node.outgoing:
                    assert edge.child.node is not None, "Linkage edge to nonexistent node"
                    if edge.tag == EdgeTags.LinkRelation:
                        assert link_relation is None, \
                            "Multiple link relations: %s, %s" % (link_relation, edge.child.node)
                        link_relation = edge.child.node
                    elif edge.tag == EdgeTags.LinkArgument:
                        link_args.append(edge.child.node)
                assert link_relation is not None, "No link relations: %s" % node
                # if len(link_args) < 2:
                #     print("Less than two link arguments for linkage %s" % node, file=sys.stderr)
                node.node = l1.add_linkage(link_relation, *link_args)
                if node.node_id:  # We are in training and we have a gold passage
                    node.node.extra["remarks"] = node.node_id  # For reference
            except AssertionError:
                if verify:
                    raise

        return passage

    def fix_terminal_tags(self, terminals):
        for terminal, orig_terminal in zip(terminals, self.terminals):
            if terminal.tag != orig_terminal.tag:
                if self.args.verbose > 2:
                    print("%s is the wrong tag for terminal: %s" % (terminal.tag, terminal.text),
                          file=sys.stderr)
                terminal.tag = orig_terminal.tag

    def topological_sort(self):
        """
        Sort self.nodes topologically, each node appearing as early as possible
        Also sort each node's outgoing and incoming edge according to the node order
        """
        levels = defaultdict(list)
        level_by_index = {}
        stack = [node for node in self.nodes if not node.outgoing]
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
                        levels[level].append(node)
                        level_by_index[node.index] = level
                else:
                    levels[0].append(node)
                    level_by_index[node.index] = 0
        self.nodes = [node for level, level_nodes in sorted(levels.items())
                      for node in sorted(level_nodes, key=lambda x: x.node_index or x.index)]
        for node in self.nodes:
            node.outgoing.sort(key=lambda x: x.child.node_index or self.nodes.index(x.child))
            node.incoming.sort(key=lambda x: x.parent.node_index or self.nodes.index(x.parent))

    def node_ratio(self):
        return (len(self.nodes) / len(self.terminals) - 1) if self.terminals else 0

    def action_ratio(self):
        return (len(self.actions) / len(self.terminals)) if self.terminals else 0

    def str(self, sep):
        return "stack: [%-20s]%sbuffer: [%s]" % (" ".join(map(str, self.stack)), sep,
                                                 " ".join(map(str, self.buffer)))

    def __str__(self):
        return self.str(" ")

    def __eq__(self, other):
        return self.stack == other.stack and self.buffer == other.buffer and \
               self.nodes == other.nodes

    def __hash__(self):
        return hash((tuple(self.stack), tuple(self.buffer), tuple(self.nodes)))
