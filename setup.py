#!/usr/bin/env python
import os
import sys
import urllib
from subprocess import run

from setuptools import setup, find_packages
from setuptools.command.install import install as _install

try:
    this_file = __file__
except NameError:
    this_file = sys.argv[0]
os.chdir(os.path.dirname(os.path.abspath(this_file)))

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

try:
    # noinspection PyPackageRequirements
    import pypandoc
    try:
        pypandoc.convert_file("README.md", "rst", outputfile="README.rst")
    except (IOError, ImportError, RuntimeError):
        pass
    long_description = pypandoc.convert_file("README.md", "rst")
except (IOError, ImportError, RuntimeError):
    long_description = ""


class install(_install):
    # noinspection PyBroadException
    def run(self):
        # Get submodules
        self.announce("Getting git submodules...")
        run(["git", "submodule", "update", "--init", "--recursive"], check=True)

        # Install requirements
        self.announce("Installing dependencies...")
        run(["pip", "--no-cache-dir", "install"] + install_requires, check=True)

        # Install AMR resource
        for filename in ("have-org-role-91-roles-v1.06.txt", "have-rel-role-91-roles-v1.06.txt",
                         "verbalization-list-v1.06.txt", "morph-verbalization-v1.01.txt"):
            out_file = os.path.join("scheme", "util", "resources", filename)
            if not os.path.exists(out_file):
                self.announce("Getting '%s'..." % filename)
                try:
                    urllib.request.urlretrieve("https://amr.isi.edu/download/lists/" + filename, out_file)
                except:
                    self.warn("Failed downloading https://amr.isi.edu/download/lists/" + filename + " to " + out_file)

        # Install actual package
        _install.run(self)


setup(name="TUPA",
      version="1.3.0",
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
      extras_require={"server": open(os.path.join("server", "requirements.txt")).read().splitlines(),
                      "viz": ["scipy", "pillow", "matplotlib"]},
      packages=find_packages() + ["src", "smatch"],
      package_dir={
          "src": os.path.join("scheme", "amr", "src"),
          "smatch": os.path.join("scheme", "smatch"),
      },
      package_data={"src": ["amr.peg"], "scheme.util": ["resources/*.txt"]},
      cmdclass={"install": install},
      entry_points={"console_scripts": ["tupa = tupa.__main__:main"]},
      )
