#!/usr/bin/env python
import urllib

import os
import pip
from setuptools import setup, find_packages
from setuptools.command.install import install as _install


install_requires = []
with open("requirements.txt") as f:
    install_requires += f.read().splitlines()
install_requires.append("dynet>=2.0")

try:
    import pypandoc
    long_description = pypandoc.convert("README.md", "rst")
except (IOError, ImportError):
    long_description = ""


class install(_install):
    def run(self):
        # Get submodules
        self.announce("Getting git submodules...")
        os.system("git submodule update --init --recursive")

        # Install requirements
        pip.main(["install", "--upgrade", "pip"])
        for package in install_requires:
            self.announce("Installing %s..." % package)
            pip.main(["install", package])
        self.announce("Dependencies installed.")

        # Install spaCy model
        try:
            __import__("en_core_web_sm")
        except ImportError:
            self.announce("Getting spaCy model...")
            import spacy
            spacy.cli.download("en")

        # Install NLTK PropBank
        try:
            from nltk.corpus import propbank
            propbank.instances()
        except LookupError:
            self.announce("Getting PropBank...")
            import nltk
            os.system("git clone https://github.com/propbank/propbank-frames " +
                      os.path.join(nltk.data.path[0], "corpora", "propbank"))

        # Install AMR resource
        for filename in ("have-org-role-91-roles-v1.06.txt", "have-rel-role-91-roles-v1.06.txt",
                         "verbalization-list-v1.06.txt", "morph-verbalization-v1.01.txt"):
            out_file = os.path.join("scheme", "util", "resources", filename)
            if not os.path.exists(out_file):
                self.announce("Getting '%s'..." % filename)
                urllib.request.urlretrieve("https://amr.isi.edu/download/lists/" + filename, out_file)

        # Install actual package
        _install.run(self)


setup(name="TUPA",
      version="1.1",
      description="Transition-based UCCA Parser",
      long_description=long_description,
      classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.6",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
      ],
      author="Daniel Hershcovich",
      author_email="danielh@cs.huji.ac.il",
      url="https://github.com/huji-nlp/tupa",
      setup_requires=["pypandoc"],
      install_requires=install_requires,
      extras_require={"server": open(os.path.join("server", "requirements.txt")).read().splitlines()},
      packages=find_packages() + ["src", "smatch"],
      package_dir={
          "src": os.path.join("scheme", "amr", "src"),
          "smatch": os.path.join("scheme", "smatch"),
      },
      package_data={"src": ["amr.peg"], "scheme.util": ["resources/*.txt"]},
      cmdclass={"install": install},
      )
