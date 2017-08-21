from ucca import convert


class SdpConverter(convert.SdpConverter):
    def __init__(self, *args, **kwargs):
        super(SdpConverter, self).__init__(*args, **kwargs)

    def modify_passage(self, passage):
        passage.extra["format"] = "sdp"


def from_sdp(lines, passage_id, split=True, mark_aux=False, *args, **kwargs):
    """Converts from parsed text in SemEval 2015 SDP format to a Passage object.

    :param lines: iterable of lines in SDP format, describing a single passage.
    :param passage_id: ID to set for passage
    :param split: split each sentence to its own passage?
    :param mark_aux: add a preceding # for labels of auxiliary edges added

    :return generator of Passage objects
    """
    del args, kwargs
    return SdpConverter(mark_aux=mark_aux).from_format(lines, passage_id, split)


def to_sdp(passage, test=False, tree=False, mark_aux=False, *args, **kwargs):
    """ Convert from a Passage object to a string in SemEval 2015 SDP format (sdp)

    :param passage: the Passage object to convert
    :param test: whether to omit the top, head, frame, etc. columns. Defaults to False
    :param tree: whether to omit columns for non-primary parents. Defaults to False
    :param mark_aux: omit edges with labels with a preceding #

    :return list of lines representing the semantic dependencies in the passage
    """
    del args, kwargs
    return SdpConverter(mark_aux=mark_aux).to_format(passage, test, tree)
