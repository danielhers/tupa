import json, pdb
import datetime, time, subprocess
import sys
import os
from bottle import route, run, error, request, static_file, debug, redirect, app, template, abort, response

import matplotlib
matplotlib.use('Agg')

PARSER_MODEL = "models/ucca-bilstm"
PARSER_TYPE = "bilstm"


def get_parser():
    if _parser is None:
        from tupa.parse import Parser
        _parser = Parser(PARSER_MODEL, PARSER_TYPE)
    return _parser
_parser = None


@route('/', method='GET')
def parser_demo():
    return static_file('demo.html', root='../../static/')

@route('/parse', method='POST')
def parse():
    from ucca.convert import from_text, to_standard
    from ucca.textutil import indent_xml
    from xml.etree.ElementTree import tostring
    text = request.forms.get("input")
    in_passage = next(from_text(text))
    out_passage = next(get_parser().parse(in_passage))
    root = to_standard(out_passage)
    xml = tostring(root).decode()
    response.headers['Content-Type'] = 'xml/application'
    return indent_xml(xml)

@route('/visualize', method='POST')
def visualize():
    from ucca.convert import from_standard
    from ucca.visualization import draw
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from io import BytesIO
    from xml.etree.ElementTree import fromstring
    from urllib.parse import quote
    from base64 import b64encode
    xml = request.body.read()
    passage = from_standard(fromstring(xml))
    print("Passage %s: %s" % (passage.ID, passage.layer("1").top_node))
    canvas = FigureCanvasAgg(plt.figure())
    draw(passage)
    image = BytesIO()
    canvas.print_png(image)
    data = b64encode(image.getvalue()).decode()
    return quote(data.rstrip('\n'))


session_opts = {
    'session.type': 'file',
    'session.cookie_expires': 60*24*60*2, #two days in seconds
    'session.data_dir': './data',
    'session.auto': True
}
sm = SessionMiddleware(app(), session_opts)

if __name__ == "__main__":
    run(host='0.0.0.0', port=5000, app=sm)
