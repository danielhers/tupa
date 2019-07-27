from .validation import Constraints, ANCHOR_LAB, ROOT_LAB


class EdgeTags:
    ParallelScene = 'h'
    Participant = 'a'
    Process = 'p'
    State = 's'
    Adverbial = 'd'
    Ground = 'g'
    Center = 'c'
    Elaborator = 'e'
    Function = 'f'
    Connector = 'n'
    Relator = 'r'
    Time = 't'
    Quantifier = 'q'
    Linker = 'l'
    Punctuation = 'u'


class UccaConstraints(Constraints):
    def __init__(self, **kwargs):
        super().__init__(multigraph=True,
                         require_implicit_childless=True, allow_orphan_terminals=False,
                         top_level_allowed={EdgeTags.ParallelScene, EdgeTags.Linker,
                                            EdgeTags.Function, EdgeTags.Ground,
                                            EdgeTags.Punctuation},
                         childless_incoming_trigger=EdgeTags.Function,
                         childless_outgoing_allowed={EdgeTags.Punctuation, ANCHOR_LAB},
                         unique_incoming={EdgeTags.Function,
                                          EdgeTags.ParallelScene, EdgeTags.Linker,
                                          EdgeTags.Connector,
                                          EdgeTags.Punctuation},
                         unique_outgoing={EdgeTags.Process, EdgeTags.State, ROOT_LAB},
                         mutually_exclusive_outgoing={EdgeTags.Process, EdgeTags.State}, **kwargs)
