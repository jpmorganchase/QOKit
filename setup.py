###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
# from setuptools import setup, Extension
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import subprocess
import os
import sys


QOKIT_PYTHON_ONLY = False
path = "./qokit/fur/c/csim/src/"

python_only = os.environ.get("QOKIT_PYTHON_ONLY")

if environment_variable_value is not None:
    QOKIT_NO_C_ENV = True


def cbuild():
    if not QOKIT_PYTHON_ONLY:
        subprocess.call(["make", "-C", path])


cbuild()
setup()
