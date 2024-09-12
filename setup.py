###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import subprocess
import os
import sys


path = "./qokit/fur/c/csim/src/"

PYTHON_ONLY = os.environ.get("QOKIT_PYTHON_ONLY") == "true"

sources = [os.path.join(path, "diagonal.c"), os.path.join(path, "fur.c"), os.path.join(path, "qaoa_fur.c")]
extensions = []
if not PYTHON_ONLY:
    extensions.append(
        Extension("simulator", sources=sources, include_dirs=[os.path.join(path, "")], extra_compile_args=["/d2FH4-"] if sys.platform == "win32" else [])
    )


def cbuild():
    if not PYTHON_ONLY:
        subprocess.call(["make", "-C", path])


class SimulatorBuild(build_ext):
    def run(self):
        cbuild()
        super().run

cbuild()
setup(ext_modules=extensions, cmdclass={"build_ext": SimulatorBuild} if sys.platform == "win32" else {}),
