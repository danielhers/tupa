import os
from argparse import ArgumentParser

from ucca import layer0
from ucca.ioutil import get_passages_with_progress_bar, write_passage


def parent(unit):
    return getattr(unit, "fparent", next(iter(unit.parents)))


def get_depth(unit):
    depth = 0
    while unit.parents:
        try:
            unit = parent(unit)
            depth += 1
        except StopIteration:
            return -1
    return depth


def get_next_terminal_lower_common_ancestor(terminal):
    try:
        next_terminal = terminal.layer.by_position(terminal.position + 1)
    except IndexError:
        return None
    unit = terminal
    while unit.parents and next_terminal not in unit.iter():
        try:
            unit = parent(unit)
        except StopIteration:
            return None
    return unit


def get_next_terminal_lower_common_ancestor_depth(terminal):
    unit = get_next_terminal_lower_common_ancestor(terminal)
    return get_depth(unit) if unit else 0


def get_next_terminal_lower_common_ancestor_category(terminal):
    unit = get_next_terminal_lower_common_ancestor(terminal)
    return getattr(unit, "ftag", None) if unit else None


def get_features(terminal):
    return dict(
        depth=get_depth(terminal),
        next_terminal_lower_common_ancestor_depth=get_next_terminal_lower_common_ancestor_depth(terminal),
        next_terminal_lower_common_ancestor_category=get_next_terminal_lower_common_ancestor_category(terminal),
    )


def hack_id(sent_id):
    """
    Replace STREUSLE sentence ID to Jakob's formatted passage ID (no "-"; zero-based)
    :param sent_id: STREUSLE sentence ID
    """
    return "%09d" % (int(sent_id.replace("-0", "")) - 1)


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    features_passage_by_id = {hack_id(passage.ID): passage for passage in get_passages_with_progress_bar(
        args.features_source_dir, desc="Reading passages for features")}
    for passage in get_passages_with_progress_bar(args.source_dir, desc="Annotating"):
        try:
            features_passage = features_passage_by_id[passage.ID]
        except KeyError as e:
            raise RuntimeError("No feature source passage found for ID=" + passage.ID) from e
        for terminal in passage.layer(layer0.LAYER_ID).all:
            try:
                features_terminal = features_passage.by_id(terminal.ID)
            except KeyError as e:
                raise RuntimeError("No terminal " + terminal.ID + " found in passage ID " + passage.ID
                                   + " from " + args.features_source_dir) from e
            terminal.extra.update(get_features(features_terminal))
        write_passage(passage, outdir=args.out_dir, verbose=False)
    print("Wrote passages to " + args.out_dir)


if __name__ == "__main__":
    argparser = ArgumentParser(description="Add structural annotations from full graph features to tokens' extra dict")
    argparser.add_argument("source_dir", help="directory to read source UCCA files from to manipulate")
    argparser.add_argument("features_source_dir", help="directory to read source UCCA files to get structure features")
    argparser.add_argument("-o", "--out-dir", default=".", help="directory to write annotated UCCA files to")
    main(argparser.parse_args())
