from ucca import convert

from .dep import DependencyConverter


class ConlluConverter(DependencyConverter, convert.ConllConverter):
    def __init__(self, *args, **kwargs):
        DependencyConverter.__init__(self, *args, tree=True, **kwargs)

    def modify_passage(self, passage):
        passage.extra["format"] = "conllu"

    def read_line(self, *args, **kwargs):
        return self.read_line_and_append(super().read_line, *args, **kwargs)


def from_conllu(lines, passage_id, split=True, return_original=False, *args, **kwargs):
    """Converts from parsed text in Universal Dependencies format to a Passage object.

    :param lines: iterable of lines in Universal Dependencies format, describing a single passage.
    :param passage_id: ID to set for passage
    :param split: split each sentence to its own passage?
    :param return_original: return triple of (UCCA passage, Universal Dependencies string, sentence ID)

    :return generator of Passage objects
    """
    del args, kwargs
    return ConlluConverter().from_format(lines, passage_id, split, return_original=return_original)


def to_conllu(passage, test=False, tree=False, constituency=False, *args, **kwargs):
    """ Convert from a Passage object to a string in Universal Dependencies format (conllu)

    :param passage: the Passage object to convert
    :param test: whether to omit the head and deprel columns. Defaults to False
    :param tree: whether to omit columns for non-primary parents. Defaults to True
    :param constituency: use UCCA conversion that introduces intermediate non-terminals

    :return list of lines representing the semantic dependencies in the passage
    """
    del args, kwargs
    return ConlluConverter(constituency=constituency).to_format(passage, test, tree)
