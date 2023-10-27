###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import os
import sys
from pathlib import Path

from qokit.classical_methods.generate_lp import generate_lp


if __name__ == "__main__":
    """
    python3 generate_lp_files.py CPLEX n_min n_max
    CPLEX: boolean if lp files will be compatible with CPLEX or not
    n_min, n_max: intergers, will generate the lp files for n_min to n_max-1 (included)

    Example: python generate_lp_files.py True 10 20
    """
    if len(sys.argv) < 4:
        raise ValueError("python3 generate_lp_files CPLEX n_min n_max")
    cplex_compatible = bool(sys.argv[1])
    n_min = int(sys.argv[2])
    n_max = int(sys.argv[3])

    cwd = Path.cwd()
    mod_path = Path(__file__).parent
    relative_path = "lp/"
    src_path = (mod_path / relative_path).resolve()

    if not os.path.isdir(src_path):
        os.makedirs(src_path)

    for n in range(n_min, n_max):
        name = f"/LABS_n{n}"
        if cplex_compatible:
            name += "_cplex"
        path = str(src_path) + name + ".lp"
        print(f"Generating LABS for n={n} in {path}")
        generate_lp(n, path, cplex_compatible=cplex_compatible)
