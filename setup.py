###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
from setuptools import setup, find_packages, Extension, Distribution
from setuptools.command.build_ext import build_ext
import subprocess
import os


environment_variable_name = "QOKIT_NO_C_ENV"
QOKIT_PYTHON_ONLY = False
QOKIT_NO_C_ENV = False  # used for tests only

environment_variable_value = os.environ.get(environment_variable_name, None)

if environment_variable_value is not None:
    QOKIT_NO_C_ENV = True

extensions = []
if not QOKIT_PYTHON_ONLY:
    extensions.append(Extension("simulator", sources=["qokit/fur/c/csim/src/*.c"], include_dirs=["simulator"]))


class SimulatorBuild(build_ext):
    def run(self):
        try:
            if not QOKIT_PYTHON_ONLY:
                if QOKIT_NO_C_ENV:
                    raise Exception("No C/C++ enviroment setup")
                subprocess.call(["make", "-C", "qokit/fur/c/csim/src"])
            super().run
        except Exception as e:
            print("No C/C++ enviroment setup to compile the C simulator. Installing Python Simulator")


with open("README.md", "r") as f:
    long_description = f.read()





""" try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
    class MyWheel(_bdist_wheel):

        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            self.root_is_pure = False

        def get_tag(self):
            python, abi, plat = _bdist_wheel.get_tag(self)
            python, abi = 'py3', 'none'
            return python, abi, plat

    class MyDistribution(Distribution):

        def __init__(self, *attrs):
            Distribution.__init__(self, *attrs)
            self.cmdclass['bdist_wheel'] = MyWheel

        def is_pure(self):
            return False

        def has_ext_modules(self):
            return True

except ImportError:
    class MyDistribution(Distribution):
        def is_pure(self):
            return False

        def has_ext_modules(self):
            return True """

try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            self.root_is_pure = False
except ImportError:
    bdist_wheel = None



setup(
    ext_modules=extensions,
    cmdclass={"build_ext": SimulatorBuild, "bdist_wheel": bdist_wheel},
    packages=find_packages(),
#    distclass=MyDistribution
) 