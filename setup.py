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
    if sys.platform in ["win32"]:
        extensions.append(Extension("simulator", sources=sources, include_dirs=[os.path.join(path, "")], extra_compile_args=["/d2FH4-"]))
    elif sys.platform.startswith("darwin"):
        extensions.append(Extension("simulator", sources=sources, include_dirs=[os.path.join(path, "")], extra_compile_args=["-Xpreprocessor"]))
    else:
        extensions.append(Extension("simulator", sources=sources, include_dirs=[os.path.join(path, "")]))


class SimulatorBuild(build_ext):
    def run(self):
        subprocess.call(["make", "clean", "-C", path])
        try:
            if not QOKIT_PYTHON_ONLY:
                if QOKIT_NO_C_ENV:
                    raise Exception("No C/C++ enviroment setup")
                subprocess.call(["make", "-C", path])
            super().run()
        except Exception as e:
            print("No C/C++ enviroment setup to compile the C simulator. Installing Python Simulator")

        # Attempt to build the max-k-xor-sat C++ backend via cmake (optional)
        if not QOKIT_PYTHON_ONLY and not QOKIT_NO_C_ENV:
            try:
                xorsat_cpp = os.path.join(".", "qokit", "max_k_xor_sat", "cpp")
                xorsat_build = os.path.join(xorsat_cpp, "build")
                os.makedirs(xorsat_build, exist_ok=True)
                subprocess.call(["cmake", "-DBUILD_SHARED_LIBS=ON", ".."], cwd=xorsat_build)
                subprocess.call(["make", "-j"], cwd=xorsat_build)
            except Exception:
                print("Optional: max-k-xor-sat C++ backend not built. Install cmake to enable.")


with open("README.md", "r") as f:
    long_description = f.read()

setup(ext_modules=extensions, cmdclass={"build_ext": SimulatorBuild})
