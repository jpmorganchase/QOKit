###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
from math import e
from pyexpat import model
from docplex.mp.model import Model
import os
import pytest

from sympy import E, assemble_partfrac_list
from qokit.classical_methods.generate_lp import generate_lp, c_k, generate_model_cplex_compatible


def test_generate_model_cplex_compatible():
    n = 5
    path = "test.lp"
    generate_lp(n, path, cplex_compatible=True)

    # Assert LP file generated succesfully
    assert os.path.exists(path)

    # Assert that LP file compatible with CPLEX
    with open(path, "r") as file:
        lp_contents = file.read()
    assert "DOcplex" in lp_contents


def test_generate_model_cplex_non_compatible():
    n = 5
    path = "test.lp"
    try:
        generate_lp(n, path, cplex_compatible=False)
    except:
        pass

    # Assert LP file generated succesfully
    assert os.path.exists(path)

    # # Assert that LP file is not compatible with CPLEX
    # with pytest.raises(Exception):
    #   pass


def test_c_k():
    n = 6
    k = 2
    print(c_k(k, n))
    expected_results = [[1, 3], [2, 4], [3, 5], [4, 6]]
    assert c_k(k, n) == expected_results
