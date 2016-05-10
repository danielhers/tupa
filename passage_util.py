import os
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
                    yield from converter(f, passage_id=base, split=Config().split)
                continue
        else:
            raise IOError("File not found: %s" % file)
        if Config().split:
            yield from convert.split2segments(passage, is_sentences=Config().args.sentences)
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


def write_passage(passage, args):
    suffix = args.format or ("pickle" if args.binary else "xml")
    outfile = args.outdir + os.path.sep + args.prefix + passage.ID + "." + suffix
    print("Writing passage '%s'..." % outfile)
    if args.format is None:
        ioutil.passage2file(passage, outfile, binary=args.binary)
    else:
        converter = convert.TO_FORMAT[args.format]
        output = "\n".join(line for line in converter(passage))
        with open(outfile, "w") as f:
            f.write(output + "\n")
