class StateEdge:
    """
    Temporary representation for graph.Edge with only relevant information for parsing
    """
    def __init__(self, parent, child, lab, attributes=None, orig_edge=None):
        self.parent = parent  # StateNode object from which this edge comes
        self.child = child  # StateNode object to which this edge goes
        self.lab = lab  # String label
        self.attributes = attributes  # dict of attribute name to value
        self.orig_edge = orig_edge
        self.src = int(parent.id)
        self.tgt = int(child.id)

    def add(self):
        assert self.parent is not self.child, "Trying to create self-loop edge on %s" % self.parent
        self.parent.add_outgoing(self)
        self.child.add_incoming(self)
        return self

    def __repr__(self):
        return StateEdge.__name__ + "(" + self.lab + ", " + repr(self.parent) + ", " + repr(self.child) + \
               ((", " + str(self.attributes)) if self.attributes else "") + ")"

    def __str__(self):
        return "%s -%s-> %s%s" % (self.parent, self.lab, self.child,
                                  ("(" + ",".join("%s=%s" % (k, v) for k, v in self.attributes.items()) + ")")
                                  if self.attributes else "")

    def __eq__(self, other):
        return other and self.parent.index == other.parent.index and self.child == other.child and \
               self.lab == other.lab

    def __hash__(self):
        return hash((self.parent.index, self.child.index, self.lab))
