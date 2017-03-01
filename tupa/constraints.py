from ucca.layer1 import EdgeTags


class Constraints:
    def __init__(self, args):
        self.args = args

    # Require all non-roots to have incoming edges
    require_connected = True

    # Require the first action to be shift, i.e., do not allow implicit children to the root
    @property
    def require_first_shift(self):
        return not self.args.implicit

    # Implicit nodes may not have children
    require_implicit_childless = True

    # Disallow terminal children to the root
    allow_root_terminal_children = False

    # Allow multiple edges (with different tags) between the same pair of nodes
    allow_multiple_edges = False

    # A unit may not have more than one outgoing edge with the same tag, if it is one of these:
    UniqueOutgoing = {
        EdgeTags.LinkRelation,
        EdgeTags.Process,
        EdgeTags.State,
    }

    def is_unique_outgoing(self, tag):
        return tag in self.UniqueOutgoing

    # A unit may not have more than one incoming edge with the same tag, if it is one of these:
    UniqueIncoming = {
        EdgeTags.Function,
        EdgeTags.Ground,
        EdgeTags.ParallelScene,
        EdgeTags.Linker,
        EdgeTags.LinkRelation,
        EdgeTags.Connector,
        EdgeTags.Punctuation,
        EdgeTags.Terminal,
    }

    def is_unique_incoming(self, tag):
        return tag in self.UniqueIncoming

    # A unit may not have more than one outgoing edge with any of these:
    MutuallyExclusiveOutgoing = {
        EdgeTags.Process,
        EdgeTags.State,
    }

    @property
    def mutually_exclusive_outgoing(self):
        return self.MutuallyExclusiveOutgoing

    # A unit may not have any children if it has any of these incoming edge tags:
    ChildlessIncoming = {
        EdgeTags.Function,
    }

    @property
    def childless_incoming(self):
        return self.ChildlessIncoming

    # A childless unit may still have these outgoing edge tags:
    ChildlessOutgoing = {
        EdgeTags.Terminal,
        EdgeTags.Punctuation,
    }

    @property
    def childless_outgoing(self):
        return self.ChildlessOutgoing

    # A unit with any outgoing edge with one of these tags is a scene:
    SceneSufficientOutgoing = {
        EdgeTags.Participant,
        EdgeTags.Process,
        EdgeTags.State,
    }

    def is_scene_sufficient_outgoing(self, tag):
        return tag in self.SceneSufficientOutgoing

    # A scene unit must have any outgoing edge with one of these tags:
    SceneNecessaryOutgoing = {
        EdgeTags.Process,
        EdgeTags.State,
    }

    def is_scene_necessary_outgoing(self, tag):
        return tag in self.SceneNecessaryOutgoing

    # A unit with any incoming edge with one of these tags is a scene:
    SceneSufficientIncoming = {
        EdgeTags.ParallelScene,
    }

    def is_scene_sufficient_incoming(self, tag):
        return tag in self.SceneSufficientIncoming

    # Outgoing edges from the root may only have these tags:
    TopLevel = {
        EdgeTags.ParallelScene,
        EdgeTags.Linker,
        EdgeTags.Function,
        EdgeTags.Ground,
        EdgeTags.Punctuation,
    }

    def is_top_level(self, tag):
        return tag in self.TopLevel

    # A linker may only have incoming edges with these tags, and must have both:
    LinkerIncoming = {
        EdgeTags.Linker,
        EdgeTags.LinkRelation,
    }

    def is_linker_incoming(self, tag):
        return tag in self.LinkerIncoming

    # Only a unit with one of these incoming tags may also have another non-remote incoming edge:
    PossibleMultipleIncoming = {
        EdgeTags.LinkArgument,
        EdgeTags.LinkRelation,
    }

    def is_possible_multiple_incoming(self, tag):
        return self.args.linkage and tag in self.PossibleMultipleIncoming
