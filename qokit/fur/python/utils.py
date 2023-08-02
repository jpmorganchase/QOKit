###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import numpy as np


def get_complex_array(sv: np.ndarray) -> np.ndarray:
    """
    create a complex NumPy array from a NumPy array or return the object as is
    if it's already a complex NumPy array
    """
    if not sv.dtype == "complex":
        sv = sv.astype("complex")
    return sv
