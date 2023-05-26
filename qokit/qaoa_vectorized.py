###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
from copy import copy
import numpy as np
import math
from typing import Optional, Sequence


class QAOAvectorizedBackendSimulator:
    # Based loosely on the ideas in
    # https://el-openqaoa.readthedocs.io/en/latest/_modules/openqaoa/backends/simulators/qaoa_vectorized.html#QAOAvectorizedBackendSimulator
    # for a given problem, requires precomputed objective function values
    def __init__(self, n_qubits: int, precomputed_energies) -> None:
        self.n_qubits = n_qubits
        self.precomputed_energies = precomputed_energies.reshape([2] * self.n_qubits)
        self.wavefn = np.ones((2**self.n_qubits,), dtype=complex) / np.sqrt(2**self.n_qubits)
        self.wavefn = self.wavefn.reshape([2] * self.n_qubits)

    def apply_diagonal(self, gamma: float) -> None:
        """
        Applies e^{-i \gamma/2 H_diag}, where H_diag is given by precomputed_energies

        Parameters
        ----------
        gamma:
            Rotation angle (evolution time)

        Returns
        -------
            None
        """
        self.wavefn *= np.exp(-1j * (gamma / 2) * self.precomputed_energies)

    def apply_hadamard(self, qubit_1: int) -> None:
        """
        Applies the Hadamard gate on ``qubit_1`` in a vectorized way. Only used when ``init_hadamard`` is true.

        Parameters
        ----------
        qubit_1:
            First qubit index to apply gate.

        Returns
        -------
            None
        """
        # TODO : Combine init_hadamard and prepend_state into one.
        # vectorized hadamard gate, for when init_hadamard = True
        wfn = copy(self.wavefn)

        slc_0 = tuple(0 if i == self.n_qubits - qubit_1 - 1 else slice(None) for i in range(self.n_qubits))
        slc_1 = tuple(1 if i == self.n_qubits - qubit_1 - 1 else slice(None) for i in range(self.n_qubits))
        wfn[slc_1] *= -1
        wfn[slc_0] += self.wavefn[slc_1]
        wfn[slc_1] += self.wavefn[slc_0]

        self.wavefn = wfn / np.sqrt(2)

    def apply_rx(self, qubit_1: int, rotation_angle: float) -> None:
        r"""
        Applies the RX($\theta$ = ``rotation_angle``) gate on ``qubit_1`` in a vectorized way.
        
        **Definition of RX($\theta$):**

        .. math::

            RX(\theta) = \exp\left(-i \frac{\theta}{2} X\right) =
            \begin{pmatrix}
                \cos{\frac{\theta}{2}}   & -i\sin{\frac{\theta}{2}} \\
                -i\sin{\frac{\theta}{2}} & \cos{\frac{\theta}{2}}
            \end{pmatrix}
            
        Parameters
        ----------
        qubit_1:
            Qubit index to apply gate.

        rotation_angle:
            Angle to be rotated.

        Returns
        -------
            None
        """

        C = np.cos(rotation_angle / 2)
        S = -1j * np.sin(rotation_angle / 2)
        wfn = (C * self.wavefn) + (S * np.flip(self.wavefn, self.n_qubits - qubit_1 - 1))

        self.wavefn = wfn

    def apply_rxy(self, q1: int, q2: int, rotation_angle: float) -> None:
        r"""
        Applies e^{-i theta (XX + YY)} on q1, q2
        Same as XXPlusYYGate in Qiskit
        https://qiskit.org/documentation/stubs/qiskit.circuit.library.XXPlusYYGate.html
        """

        x = copy(self.wavefn.flatten())
        theta = rotation_angle / 2

        if q1 > q2:
            q1, q2 = q2, q1

        num_states = len(x)
        num_groups = num_states // 4

        mask1 = (1 << q1) - 1
        mask2 = (1 << (q2 - 1)) - 1
        maskm = mask1 ^ mask2
        mask2 ^= (num_states - 1) >> 2

        wa = math.cos(theta)
        wb = -1j * math.sin(theta)

        for i in range(num_groups):
            i0 = (i & mask1) + ((i & maskm) << 1) + ((i & mask2) << 2)
            ia = i0 + (1 << q1)
            ib = i0 + (1 << q2)
            x[ia], x[ib] = wa * x[ia] + wb * x[ib], wb * x[ia] + wa * x[ib]

        self.wavefn = x.reshape([2] * self.n_qubits)

    def apply_qaoa_circuit(self, beta: Sequence, gamma: Sequence) -> None:
        """Applies a QAOA circuit
        Parameterization matches the one used in qaoa_qiskit.py

        Parameters
        ----------
        beta : list-like
            QAOA parameter beta
        gamma : list-like
            QAOA parameter gamma

        Returns
        -------
            None
        """
        assert len(beta) == len(gamma)
        for beta_j, gamma_j in zip(beta, gamma):
            self.apply_diagonal(gamma_j)
            for idx in range(self.n_qubits):
                self.apply_rx(idx, 2 * beta_j)


def get_qaoa_statevector(beta: Sequence, gamma: Sequence, N: Optional[int] = None, precomputed_energies=None) -> np.ndarray:
    """Get QAOA statevector using the vectorized simulator

    Parameters
    ----------
    beta : list-like
        QAOA parameter beta
    gamma : list-like
        QAOA parameter gamma


    Returns
    -------
    sv : np.array
        Vector of amplitudes
    """
    sim = QAOAvectorizedBackendSimulator(N, precomputed_energies)
    sim.apply_qaoa_circuit(beta, gamma)
    return sim.wavefn.flatten()
