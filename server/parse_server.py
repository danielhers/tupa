import json
import os
from base64 import b64encode
from io import BytesIO, StringIO
from urllib.parse import quote

import flask_assets
import jinja2
import matplotlib
import pydot
from flask import Flask, render_template, Response, request
from flask_compress import Compress
from main import read_graphs
from webassets import Environment as AssetsEnvironment
from webassets.ext.jinja2 import AssetsExtension

from tupa.parse import Parser

matplotlib.use("Agg")

SCRIPT_DIR = os.path.dirname(__file__)

app = Flask(__name__)
assets = flask_assets.Environment()
assets.init_app(app)
assets_env = AssetsEnvironment("./static/", "/static")
jinja_environment = jinja2.Environment(
    autoescape=True,
    loader=jinja2.FileSystemLoader(os.path.join(SCRIPT_DIR, "templates")),
    extensions=[AssetsExtension])
jinja_environment.assets_environment = assets_env
Compress(app)

app.parser = None
PARSER_MODEL = os.getenv("PARSER_MODEL", os.path.join(SCRIPT_DIR, "..", "models/bilstm"))


def get_parser():
    if app.parser is None:
        print("Initializing parser...")
        print("PARSER_MODEL=" + PARSER_MODEL)
        app.parser = Parser(PARSER_MODEL)
    return app.parser


@app.route("/")
def parser_demo():
    return render_template("demo.html")


@app.route("/parse", methods=["POST"])
def parse():
    text = request.values["input"]
    print("Parsing text: '%s'" % text)
    out = next(get_parser().parse(text))[0]
    return Response(json.dumps(out.encode(), indent=None, ensure_ascii=False),
                    headers={"Content-Type": "application/json"})


@app.route("/visualize", methods=["POST"])
def visualize():
    json = request.get_data()
    graph = read_graphs(json, format="mrp")[0]
    print("Visualizing graph %s: %s" % (graph.id, json))
    s = StringIO()
    graph.dot(s)
    (graph,) = pydot.graph_from_dot_data(s.getvalue())
    image = BytesIO()
    graph.write_png(image)
    data = b64encode(image.getvalue()).decode()
    return Response(quote(data.rstrip("\n")))


CONTENT_TYPES = {"json": "application/json"}


session_opts = {
    "session.type": "file",
    "session.cookie_expires": 60 * 24 * 60 * 2,  # two days in seconds
    "session.data_dir": "./data",
    "session.auto": True
}

if __name__ == "__main__":
    app.run(debug=True, host=os.getenv("IP", "ucca"), port=int(os.getenv("PORT", 5001)))
