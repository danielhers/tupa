#!/usr/bin/env python

import os
from distutils.core import setup

setup(name="TUPA",
      version="1.0",
      description="Transition-based UCCA Parser",
      author="Daniel Hershcovich",
      author_email="danielh@cs.huji.ac.il",
      url="http://www.cs.huji.ac.il/~oabend/ucca.html",
      packages=["scheme", "constraint", "conversion", "evaluation", "util", "src", "smatch",
                "tupa", "classifiers", "nn", "linear", "features", "states"],
      package_dir={
          "scheme": "scheme",
          "constraint": os.path.join("scheme", "constraint"),
          "conversion": os.path.join("scheme", "conversion"),
          "evaluation": os.path.join("scheme", "evaluation"),
          "util": os.path.join("scheme", "util"),
          "src": os.path.join("scheme", "amr", "src"),
          "smatch": os.path.join("scheme", "smatch"),
          "tupa": "tupa",
          "classifiers": os.path.join("tupa", "classifiers"),
          "linear": os.path.join("tupa", "classifiers", "linear"),
          "nn": os.path.join("tupa", "classifiers", "nn"),
          "features": os.path.join("tupa", "features"),
          "states": os.path.join("tupa", "states"),
      },
      package_data={"src": ["amr.peg"]},
      )
