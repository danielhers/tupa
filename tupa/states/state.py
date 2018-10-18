from collections import deque

from semstr.constraints import Constraints, Direction
from semstr.util.amr import LABEL_ATTRIB
from semstr.validation import CONSTRAINTS
from ucca import core, layer0, layer1
from ucca.layer0 import NodeTags
from ucca.layer1 import EdgeTags

from .edge import Edge
from .node import Node
from ..action import Actions
from ..config import Config


class InvalidActionError(AssertionError):
    def __init__(self, *args, is_type=False):
        super().__init__(*args)
        self.is_type = is_type


class State:
    """
    The parser's state, responsible for applying actions and creating the final Passage
    :param passage: a Passage object to get the tokens from, and everything else if training
    """
    def __init__(self, passage):
        self.args = Config().args
        self.constraints = CONSTRAINTS.get(passage.extra.get("format"), Constraints)(implicit=self.args.implicit)
        self.log = []
        self.finished = False
        self.passage = passage
        l0, l1 = [self.get_layer(passage, l) for l in (layer0, layer1)]
        self.labeled = any(n.outgoing or n.attrib.get(LABEL_ATTRIB) for n in l1.all)
        self.terminals = [Node(i, orig_node=t, root=passage, text=t.text, paragraph=t.paragraph, tag=t.tag)
                          for i, t in enumerate(l0.all, start=1)]
        self.stack = []
        self.buffer = deque()
        self.nodes = []
        self.heads = set()
        self.need_label = None  # If we are waiting for label_node() to be called, which node is to be labeled by it
        self.root = self.add_node(orig_node=l1.heads[0], is_root=True)  # Root is not in the buffer
        self.stack.append(self.root)
        self.buffer += self.terminals
        self.nodes += self.terminals
        self.actions = []  # History of applied actions
        self.type_validity_cache = {}

    @staticmethod
    def get_layer(passage, layer):
        try:
            return passage.layer(layer.LAYER_ID)
        except KeyError as e:
            raise IOError("Passage %s is missing layer %s" % (passage.ID, layer.LAYER_ID)) from e

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

        def _check_possible_parent(node, t):
            self.check(node.text is None, message and "Terminals may not have children: %s" % node.text, is_type=True)
            if self.args.constraints and t is not None:
                for rule in self.constraints.tag_rules:
                    violation = rule.violation(node, t, Direction.outgoing, message=message)
                    self.check(violation is None, violation)
                self.check(self.constraints.allow_parent(node, t),
                           message and "%s may not be a '%s' parent (currently %s)" % (
                               node, t, ", ".join(map(str, node.outgoing)) or "childless"))
            self.check(not self.constraints.require_implicit_childless or not node.implicit,
                       message and "Implicit nodes may not have children: %s" % s0, is_type=True)

        def _check_possible_child(node, t):
            self.check(node is not self.root, message and "Root may not have parents", is_type=True)
            if self.args.constraints and t is not None:
                self.check(not t or (node.text is None) != (t == EdgeTags.Terminal),
                           message and "Edge tag must be %s iff child is terminal, but node %s has edge tag %s" %
                           (EdgeTags.Terminal, node, t))
                for rule in self.constraints.tag_rules:
                    violation = rule.violation(node, t, Direction.incoming, message=message)
                    self.check(violation is None, violation)
                self.check(self.constraints.allow_child(node, t),
                           message and "%s may not be a '%s' child (currently %s, %s)" % (
                               node, t, ", ".join(map(str, node.incoming)) or "parentless",
                               ", ".join(map(str, node.outgoing)) or "childless"))
            self.check(self.constraints.possible_multiple_incoming is None or t is None or
                       action.remote or t in self.constraints.possible_multiple_incoming or
                       all(e.remote or e.tag in self.constraints.possible_multiple_incoming for e in node.incoming),
                       message and "Multiple non-remote '%s' parents not allowed for %s" % (t, node))

        def _check_possible_edge(p, c, t):
            _check_possible_parent(p, t)
            _check_possible_child(c, t)
            if self.args.constraints and t is not None:
                if p is self.root:
                    self.check(self.constraints.top_level_allowed is None or not t or
                               t in self.constraints.top_level_allowed, message and "Root may not have %s edges" % t)
                else:
                    self.check(self.constraints.top_level_only is None or
                               t not in self.constraints.top_level_only, message and "Only root may have %s edges" % t)
            self.check(self.constraints.allow_root_terminal_children or p is not self.root or c.text is None,
                       message and "Terminal child '%s' for root" % c, is_type=True)
            if self.constraints.multigraph:  # Nodes may be connected by more than one edge
                edge = Edge(p, c, t, remote=action.remote)
                self.check(self.constraints.allow_edge(edge), message and "Edge not allowed: %s (currently: %s)" % (
                               edge, ", ".join(map(str, p.outgoing)) or "childless"))
            else:  # Simple graph, i.e., no more than one edge between the same pair of nodes
                self.check(c not in p.children, message and "%s is already %s's child" % (c, p), is_type=True)
            self.check(p not in c.descendants, message and "Detected cycle by edge: %s->%s" % (p, c), is_type=True)

        def _check_possible_label():
            self.check(self.args.node_labels, message and "Node labels disabled", is_type=True)
            try:
                node = self.stack[-action.tag]
            except IndexError:
                node = None
            self.check(node is not None, message and "Labeling invalid node %s when stack size is %d" % (
                action.tag, len(self.stack)))
            self.check(not node.labeled, message and "Labeling already-labeled node: %s" % node, is_type=True)
            self.check(node.text is None, message and "Terminals do not have labels: %s" % node, is_type=True)

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
            elif action.is_type(Actions.Label):
                _check_possible_label()
            else:   # Unary actions
                self.check(self.stack, message and "%s with empty stack" % action, is_type=True)
                s0 = self.stack[-1]
                if action.is_type(Actions.Reduce):
                    if s0 is self.root:
                        self.check(self.root.labeled or not self.args.node_labels,
                                   message and "Reducing root without label", is_type=True)
                    elif not s0.text:
                        self.check(not self.args.require_connected or s0.is_linkage or s0.incoming,
                                   message and "Reducing parentless non-terminal %s" % s0, is_type=True)
                        self.check(not self.constraints.required_outgoing or
                                   s0.outgoing_tags.intersection((EdgeTags.Terminal, EdgeTags.Punctuation, "")) or
                                   s0.outgoing_tags.intersection(self.constraints.required_outgoing),
                                   message and "Reducing non-terminal %s without %s edge" % (
                                       s0, self.constraints.required_outgoing), is_type=True)
                    self.check(not self.args.node_labels or s0.text or s0.labeled,
                               message and "Reducing non-terminal %s without label" % s0, is_type=True)
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
                    pct = self.get_parent_child_tag(action)
                    self.check(pct, message and "%s with len(stack) = %d" % (action, len(self.stack)), is_type=True)
                    parent, child, tag = pct
                    if parent is None:
                        _check_possible_child(child, tag)
                        _check_possible_node()
                    elif child is None:
                        _check_possible_parent(parent, tag)
                        _check_possible_node()
                    else:  # Binary actions
                        _check_possible_edge(parent, child, tag)

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
            valid = self.constraints.allow_label(self.need_label, label)
            self.check(valid, message and "May not label %s as %s: %s" % (self.need_label, label, valid))

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
        pct = self.get_parent_child_tag(action)
        if pct:
            parent, child, tag = pct
            if parent is None:
                parent = action.node = self.add_node(orig_node=action.orig_node)
            if child is None:
                child = action.node = self.add_node(orig_node=action.orig_node, implicit=True)
            action.edge = self.add_edge(Edge(parent, child, tag, remote=action.remote))
            if action.node:
                self.buffer.appendleft(action.node)
        elif action.is_type(Actions.Shift):  # Push buffer head to stack; shift buffer
            self.stack.append(self.buffer.popleft())
        elif action.is_type(Actions.Label):
            self.need_label = self.stack[-action.tag]  # The parser is responsible to choose a label and set it
        elif action.is_type(Actions.Reduce):  # Pop stack (no more edges to create with this node)
            self.stack.pop()
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
        if self.args.use_gold_node_labels:
            self.need_label = node  # Labeled the node as soon as it is created rather than applying a LABEL action
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
            b0 = self.buffer[0]
            if self.stack and (b0.text is not None or b0.swap_index <= len(self.nodes)):
                s0 = self.stack[-1]
                return (s0.swap_index + b0.swap_index) / 2
        return None

    def add_edge(self, edge):
        edge.add()
        self.heads.discard(edge.child)
        self.log.append("edge: %s" % edge)
        return edge
    
    PARENT_CHILD = (
        ((Actions.LeftEdge, Actions.LeftRemote), (-1, -2)),
        ((Actions.RightEdge, Actions.RightRemote), (-2, -1)),
        ((Actions.Node, Actions.RemoteNode), (None, -1)),
        ((Actions.Implicit,), (-1, None)),
    )

    def get_parent_child_tag(self, action):
        try:
            for types, indices in self.PARENT_CHILD:
                if action.is_type(*types):
                    parent, child = [None if i is None else self.stack[i] for i in indices]
                    break
            else:
                return None
            return parent, child, (EdgeTags.Terminal if child and child.text else
                                   EdgeTags.Punctuation if child and child.children and all(
                                       c.tag == NodeTags.Punct for c in child.children)
                                   else action.tag)  # In unlabeled parsing, keep a valid graph
        except IndexError:
            return None

    def label_node(self, label):
        self.need_label.label = label
        self.need_label.labeled = True
        self.log.append("label: %s" % self.need_label)
        self.type_validity_cache = {}
        self.need_label = None

    def create_passage(self, verify=True, **kwargs):
        """
        Create final passage from temporary representation
        :param verify: fail if this results in an improper passage
        :return: core.Passage created from self.nodes
        """
        Config().print("Creating passage %s from state..." % self.passage.ID, level=2)
        passage = core.Passage(self.passage.ID)
        passage_format = kwargs.get("format") or self.passage.extra.get("format")
        if passage_format:
            passage.extra["format"] = passage_format
        self.passage.layer(layer0.LAYER_ID).copy(passage)
        l0 = passage.layer(layer0.LAYER_ID)
        l1 = layer1.Layer1(passage)
        self.root.node = l1.heads[0]
        if self.args.node_labels:
            self.root.set_node_label()
        if self.labeled:  # We have a reference passage
            self.root.set_node_id()
        Node.attach_nodes(l0, l1, self.nodes, self.labeled, self.args.node_labels, verify)
        return passage

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
