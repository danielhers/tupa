from ucca import convert

from .dep import DependencyConverter


class SdpConverter(DependencyConverter, convert.SdpConverter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def modify_passage(self, passage):
        passage.extra["format"] = "sdp"

    def edges_for_orphan(self, top):
        return [self.Edge(0, self.TOP, False)] if top else []

    def read_line(self, line, previous_node):
        self.lines_read.append(line)
        try:
            return super().read_line(line, previous_node)
        except ValueError as e:
            raise ValueError("Failed reading line:\n" + line) from e


def from_sdp(lines, passage_id, split=True, mark_aux=False, return_original=False, *args, **kwargs):
    """Converts from parsed text in SemEval 2015 SDP format to a Passage object.

    :param lines: iterable of lines in SDP format, describing a single passage.
    :param passage_id: ID to set for passage
    :param split: split each sentence to its own passage?
    :param mark_aux: add a preceding # for labels of auxiliary edges added
    :param return_original: return triple of (UCCA passage, SDP string, sentence ID)

    :return generator of Passage objects
    """
    del args, kwargs
    return SdpConverter(mark_aux=mark_aux).from_format(lines, passage_id, split, return_original=return_original)


def to_sdp(passage, test=False, tree=False, mark_aux=False, constituency=False, *args, **kwargs):
    """ Convert from a Passage object to a string in SemEval 2015 SDP format (sdp)

    :param passage: the Passage object to convert
    :param test: whether to omit the top, head, frame, etc. columns. Defaults to False
    :param tree: whether to omit columns for non-primary parents. Defaults to False
    :param mark_aux: omit edges with labels with a preceding #
    :param constituency: use UCCA conversion that introduces intermediate non-terminals

    :return list of lines representing the semantic dependencies in the passage
    """
    del args, kwargs
    return SdpConverter(mark_aux=mark_aux, constituency=constituency).to_format(passage, test, tree)
