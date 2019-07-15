from .validation import Constraints, ANCHOR_LAB


class EdgeTags:
    ParallelScene = 'H'
    Participant = 'A'
    Process = 'P'
    State = 'S'
    Adverbial = 'D'
    Ground = 'G'
    Center = 'C'
    Elaborator = 'E'
    Function = 'F'
    Connector = 'N'
    Relator = 'R'
    Time = 'T'
    Quantifier = 'Q'
    Linker = 'L'
    Punctuation = 'U'


class UccaConstraints(Constraints):
    def __init__(self, **kwargs):
        super().__init__(require_implicit_childless=True, allow_orphan_terminals=False,
                         allow_root_terminal_children=False,
                         top_level_allowed={EdgeTags.ParallelScene, EdgeTags.Linker,
                                            EdgeTags.Function, EdgeTags.Ground,
                                            EdgeTags.Punctuation},
                         childless_incoming_trigger=EdgeTags.Function,
                         childless_outgoing_allowed={EdgeTags.Punctuation, ANCHOR_LAB},
                         unique_incoming={EdgeTags.Function,
                                          EdgeTags.ParallelScene, EdgeTags.Linker,
                                          EdgeTags.Connector,
                                          EdgeTags.Punctuation},
                         unique_outgoing={EdgeTags.Process, EdgeTags.State},
                         mutually_exclusive_outgoing={EdgeTags.Process, EdgeTags.State}, **kwargs)
