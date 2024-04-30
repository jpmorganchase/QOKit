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

sources = [os.path.join(path, "diagonal.c"), os.path.join(path, "fur.c"), os.path.join(path, "qaoa_fur.c")]
extensions = []
if python_only is None:
    extensions.append(
        Extension("simulator", sources=sources, include_dirs=[os.path.join(path, "")], extra_compile_args=["/d2FH4-"] if sys.platform == "win32" else [])
    )

def cbuild():
    if python_only is None:
        subprocess.call(["make", "-C", path])


cbuild()
setup(ext_modules=extensions)
