import os
from itertools import groupby
from operator import not_
from xml.etree.ElementTree import ParseError

from parsing.config import Config
from ucca import core, convert, ioutil


def read_passages(files):
    """
    :param files: iterable of files or Passage objects
    :return: generator of passages from all files given
    """
    for file in files:
        if isinstance(file, core.Passage):  # Not really a file, but a Passage
            passage = file
        elif os.path.exists(file):  # A file
            try:
                passage = ioutil.file2passage(file)  # XML or binary format
            except (IOError, ParseError):  # Failed to read as passage file
                base, ext = os.path.splitext(os.path.basename(file))
                converter = convert.FROM_FORMAT.get(ext.lstrip("."), convert.from_text)
                with open(file) as f:
                    passage = next(converter(f, passage_id=base))
        else:
            raise IOError("File not found: %s" % file)
        if Config().split:
            yield from convert.split2segments(
                passage, is_sentences=Config().sentences, remarks=True)
        else:
            yield passage


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