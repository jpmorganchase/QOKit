###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import subprocess
import os


path = "./qokit/fur/c/csim/src/"

python_only = os.environ.get("QOKIT_PYTHON_ONLY")


def cbuild():
    if python_only is None:
        subprocess.call(["make", "-C", path])


cbuild()
setup(ext_modules=[Extension(name="your.external.module", sources=[])])
