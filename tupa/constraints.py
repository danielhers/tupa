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
    unique_outgoing = {
        EdgeTags.LinkRelation,
        EdgeTags.Process,
        EdgeTags.State,
    }

    # A unit may not have more than one incoming edge with the same tag, if it is one of these:
    unique_incoming = {
        EdgeTags.Function,
        EdgeTags.Ground,
        EdgeTags.ParallelScene,
        EdgeTags.Linker,
        EdgeTags.LinkRelation,
        EdgeTags.Connector,
        EdgeTags.Punctuation,
        EdgeTags.Terminal,
    }

    # A unit may not have more than one outgoing edge with any of these:
    mutually_exclusive_outgoing = {
        EdgeTags.Process,
        EdgeTags.State,
    }

    # A unit may not have any children if it has any of these incoming edge tags:
    childless_incoming = {
        EdgeTags.Function,
    }

    # A childless unit may still have these outgoing edge tags:
    childless_outgoing = {
        EdgeTags.Terminal,
        EdgeTags.Punctuation,
    }

    # A unit with any outgoing edge with one of these tags is a scene:
    scene_sufficient_outgoing = {
        EdgeTags.Participant,
        EdgeTags.Process,
        EdgeTags.State,
    }

    # A scene unit must have any outgoing edge with one of these tags:
    scene_necessary_outgoing = {
        EdgeTags.Process,
        EdgeTags.State,
    }

    # A unit with any incoming edge with one of these tags is a scene:
    scene_sufficient_incoming = {
        EdgeTags.ParallelScene,
    }

    # Outgoing edges from the root may only have these tags:
    top_level = {
        EdgeTags.ParallelScene,
        EdgeTags.Linker,
        EdgeTags.Function,
        EdgeTags.Ground,
        EdgeTags.Punctuation,
    }

    # A linker may only have incoming edges with these tags, and must have both:
    linker_incoming = {
        EdgeTags.Linker,
        EdgeTags.LinkRelation,
    }

    # Only a unit with one of these incoming tags may also have another non-remote incoming edge:
    @property
    def possible_multiple_incoming(self):
        return {EdgeTags.LinkArgument, EdgeTags.LinkRelation} if self.args.linkage else ()
