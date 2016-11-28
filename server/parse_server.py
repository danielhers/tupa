from xml.etree.ElementTree import tostring

from flask import Flask, request, Response, render_template

from parsing.config import Config
from parsing.parse import Parser
from ucca.convert import from_text, to_standard
from ucca.textutil import indent_xml

app = Flask(__name__)


@app.before_first_request
def initialize():
    config = Config()
    app.parser = Parser(config.args.model, config.args.classifier)
    app.logger.info("Initialized parser")


@app.route("/parse", methods=["GET", "POST"])
def parse():
    text = request.values["input"]
    app.logger.info("Parsing text '%s'" % text)
    in_passage = next(from_text(text))
    out_passage = next(app.parser.parse(in_passage))
    root = to_standard(out_passage)
    xml = tostring(root).decode()
    response = indent_xml(xml)
    return Response(response, mimetype="text/xml")


@app.route("/")
def index():
    return render_template("demo.html")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
