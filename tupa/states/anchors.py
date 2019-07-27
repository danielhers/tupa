def expand_anchors(anchors):
    """ Convert {from, to} dict to set of integers with the full ranges """
    return set.union(*[set(range(x["from"] if x["from"] < x["to"] else x["from"] - 1, x["to"]))
                       for x in anchors]) if anchors else set()


# noinspection PyTypeChecker
def compress_anchors(anchors):
    """
    Convert set of integers back to {from, to} dict
    >>> compress_anchors({13})
    [{'from': 13, 'to': 14}]
    >>> compress_anchors({0, 1, 2, 3, 4, 5})
    [{'from': 0, 'to': 6}]
    >>> compress_anchors({13, 18})
    [{'from': 13, 'to': 14}, {'from': 18, 'to': 19}]
    >>> compress_anchors({0, 1, 2, 4, 5})
    [{'from': 0, 'to': 3}, {'from': 4, 'to': 6}]
    """
    ranges = []
    start = end = None
    for anchor in sorted(anchors) + [None]:
        if start is not None and (anchor is None or anchor > end):  # Finished a range
            ranges.append({"from": start, "to": end})
            start = None
        if start is None:  # Starting a new range
            start = anchor
        if anchor is not None:
            end = anchor + 1
    return ranges
