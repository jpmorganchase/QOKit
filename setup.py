from setuptools import setup, find_packages, Extension
from sys import stderr, stdout, exit
from setuptools.command.build_ext import build_ext
import subprocess


simulator = Extension("simulator", sources=["qokit/fur/c/csim/src/*.c"], include_dirs=["simulator"])


class SimulatorBuild(build_ext):
    def run(self):
        subprocess.call(["make", "-C", "qokit/fur/c/csim/src"])
        super().run


with open("README.md", "r") as f:
    long_description = f.read()


setup(
    ext_modules=[simulator],
    cmdclass={"build_ext": SimulatorBuild},
    packages=find_packages(),
    install_requires=open("requirements.txt").readlines(),
)
