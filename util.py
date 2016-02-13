import os
from itertools import groupby
from operator import not_
from xml.etree.ElementTree import ParseError

from parsing.config import Config
from ucca import core, convert, ioutil


def read_passage(passage):
    """
    Read a passage given in any format
    :param passage: either a core.Passage, a file, or a list of list of strings (paragraphs, words)
    :return: a core.Passage and its ID if given a Passage or file, or else the given list of lists
    """
    if isinstance(passage, core.Passage):
        passage_id = passage.ID
    elif os.path.exists(passage):  # a file
        try:
            passage = ioutil.file2passage(passage)  # XML or binary format
            passage_id = passage.ID
        except (IOError, ParseError):
            passage_id, ext = os.path.splitext(os.path.basename(passage))
            converter = convert.CONVERTERS.get(ext.lstrip("."))
            with open(passage) as f:
                if converter is None:  # Simple text file
                    passage = [[token for line in group for token in line.split()]
                               for is_sep, group in groupby(f.read.splitlines(), not_)
                               if not is_sep]
                else:  # Known extension, convert to passage
                    converter, _ = converter
                    passage = next(converter(f, passage_id))
    else:
        raise IOError("File not found: %s" % passage)
    return passage, passage_id


def read_passages(files):
    for file in files:
        passage, i = read_passage(file)
        if Config().split:
            segments = convert.split2segments(passage, is_sentences=Config().sentences,
                                              remarks=True)
            for j, segment in enumerate(segments):
                yield (segment, "%s_%d" % (i, j))
        else:
            yield (passage, i)


def read_files_and_dirs(files_and_dirs):
    """
    :param files_and_dirs: iterable of files and/or directories to look in
    :return: generator of passages from all files given,
             plus any files directly under any directory given
    """
    files = list(files_and_dirs)
    files += [os.path.join(d, f) for d in files if os.path.isdir(d) for f in os.listdir(d)]
    files = [f for f in files if not os.path.isdir(f)]
    return read_passages(files) if files else ()


def write_passage(passage, outdir, prefix, binary, verbose):
    suffix = ".pickle" if binary else ".xml"
    outfile = outdir + os.path.sep + prefix + passage.ID + suffix
    if verbose:
        print("Writing passage '%s'..." % outfile)
    ioutil.passage2file(passage, outfile, binary=binary)