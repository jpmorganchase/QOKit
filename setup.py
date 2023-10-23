###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import subprocess
import os
import sys


environment_variable_name = "QOKIT_NO_C_ENV"
QOKIT_PYTHON_ONLY = False
QOKIT_NO_C_ENV = False  # used for tests only

environment_variable_value = os.environ.get(environment_variable_name, None)

if environment_variable_value is not None:
    QOKIT_NO_C_ENV = True

path = "./qokit/fur/c/csim/src/"

sources = [os.path.join(path, "diagonal.c"), os.path.join(path, "fur.c"), os.path.join(path, "qaoa_fur.c")]

extensions = []
if not QOKIT_PYTHON_ONLY:
    extensions.append(
        Extension("simulator", sources=sources, include_dirs=[os.path.join(path, "")], extra_compile_args=["/d2FH4-"] if sys.platform == "win32" else [])
    )


class SimulatorBuild(build_ext):
    def run(self):
        try:
            if not QOKIT_PYTHON_ONLY:
                if QOKIT_NO_C_ENV:
                    raise Exception("No C/C++ enviroment setup")
                subprocess.call(["make", "-C", path])
            super().run
        except Exception as e:
            print("No C/C++ enviroment setup to compile the C simulator. Installing Python Simulator")


with open("README.md", "r") as f:
    long_description = f.read()


setup(ext_modules=extensions, cmdclass={"build_ext": SimulatorBuild} if sys.platform == "win32" else {}),
