class Edge:
    """
    Temporary representation for core.Edge with only relevant information for parsing
    """
    def __init__(self, parent, child, tag, attributes=None):
        self.parent = parent  # Node object from which this edge comes
        self.child = child  # Node object to which this edge goes
        self.tag = tag  # String tag
        self.attributes = attributes  # dict of attribute name to value

    def add(self):
        assert self.parent is not self.child, "Trying to create self-loop edge on %s" % self.parent
        self.parent.add_outgoing(self)
        self.child.add_incoming(self)

    def __repr__(self):
        return Edge.__name__ + "(" + self.tag + ", " + repr(self.parent) + ", " + repr(self.child) +\
               ((", " + str(self.attributes)) if self.attributes else "") + ")"

    def __str__(self):
        return "%s -%s-> %s%s" % (self.parent, self.tag, self.child,
                                  (" (" + ",".join("%s=%s" % (k, v) for k, v in self.attributes.items()) + ")")
                                  if self.attributes else "")

    def __eq__(self, other):
        return other and self.parent.index == other.parent.index and self.child == other.child and \
               self.tag == other.tag

    def __hash__(self):
        return hash((self.parent.index, self.child.index, self.tag))
