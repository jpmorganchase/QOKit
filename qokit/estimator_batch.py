from __future__ import annotations
import numpy as np
from qiskit.primitives import Estimator

def batched_expectation(
    est: Estimator,
    circuit,
    observable,
    thetas: np.ndarray,          # shape (B, 2p)
) -> np.ndarray:
    """
    Evaluate ⟨H⟩ for *B* parameter vectors in **one** Estimator call.

    Returns
    -------
    np.ndarray
        1-D array of length B with expectation values.
    """
    B = len(thetas)
    return est.run(
        circuits   =[circuit] * B,
        observables=[observable] * B,
        parameter_values=thetas,
    ).result().values.real
