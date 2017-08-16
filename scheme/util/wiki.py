import os
import spotlight
from functools import partial
from spotlight import SpotlightException
from ucca import layer1

from scheme.util.amr import NAME, WIKI, LABEL_ATTRIB

spot = partial(spotlight.annotate,
               os.environ.get("SPOTLIGHT_ADDRESS", "http://model.dbpedia-spotlight.org/en/annotate"),
               confidence=float(os.environ.get("SPOTLIGHT_CONFIDENCE", 0.3)))


def wikify_node(node, name):
    try:
        text = " ".join(t.text for t in node.get_terminals()) or \
               " ".join(filter(None, (n.attrib.get(LABEL_ATTRIB) for n in name.children))).replace('"', "")
        spots = spot(text) if text else ()
        return '"%s"' % spots[0]["URI"].replace("http://dbpedia.org/resource/", "")
    except (ValueError, IndexError, SpotlightException):
        return "-"


def wikify(passage):
    l1 = passage.layer(layer1.LAYER_ID)
    for node in l1.all:
        name = wiki = None
        for edge in node:
            if edge.tag == NAME:
                name = edge
            elif edge.tag == WIKI:
                wiki = edge
        if wiki is not None:
            node.remove(wiki)
            if name is not None:
                l1.add_fnode(node, WIKI).attrib[LABEL_ATTRIB] = wikify_node(node, name.child)
