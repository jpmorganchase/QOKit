#
# Script for setting up juliacall in python, and for installing the julia
# packages needed for our proxy implementations.
#

import subprocess
import sys

#Install julicall using pip.
package_name = "juliacall"
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
    print(f"Successfully installed {package_name}")
except subprocess.CalledProcessError as e:
    print(f"Failed to install {package_name}. Error: {e}")

from juliacall import Main as jl
jl.seval("""
import Pkg
Pkg.add("Distributions")
Pkg.add("BenchmarkTools")
Pkg.add("TimerOutputs")
Pkg.add("PythonCall")
""")