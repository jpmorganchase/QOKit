###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################

import numpy as np
import pytest

from qokit.qaoa_objective import get_qaoa_objective

@pytest.mark.parametrize("mixer,expected_khot", [
    ("x", None),
    ("xy", 2),
    ("swap", 2)
])
def test_khot_flag_for_mixers(mixer, expected_khot):
    N = 4
    K = 2
    p = 1

    terms = [
        (+1.0, [0, 1]),
        (-0.5, [2, 3])
    ]

    if expected_khot is None:
        f = get_qaoa_objective(
            N=N,
            terms=terms,
            mixer=mixer,
            k_hot=None
        )
    else:
        f = get_qaoa_objective(
            N=N,
            terms=terms,
            mixer=mixer,
            k_hot=K
        )
    
    theta = np.zeros(2 * p)
    val = f(theta)

    # Check that the function returns a float (runs without error)
    assert isinstance(val, float)

    # Print confirmation for debug
    print(f"Mixer={mixer} ran successfully with k_hot={expected_khot}")
