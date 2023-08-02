###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
from __future__ import annotations
import typing
import numpy as np
from abc import ABC, abstractmethod
import sys

# Terms are a list of tuples (coeff, [qubit indices])
# Run this only for python> 3.9
if sys.version_info >= (3, 10):
    from collections.abc import Sequence

    TermsType = Sequence[tuple[float, Sequence[int]]]
    CostsType = Sequence[float] | np.ndarray
    ParamType = Sequence[float] | np.ndarray
else:
    from typing import Sequence

    sys.modules["collections.abc"].Sequence = typing.Sequence
    TermsType = Sequence[typing.Tuple[float, Sequence[int]]]
    CostsType = typing.Union[Sequence[float], np.ndarray]
    ParamType = typing.Union[Sequence[float], np.ndarray]
    from collections.abc import Sequence


class QAOAFastSimulatorBase(ABC):
    """
    Base class for QAOA simulator
    """

    def __init__(
        self,
        n_qubits: int,
        costs: CostsType | None = None,
        terms: TermsType | None = None,
    ) -> None:
        """
        Create the simulator.
        Either costs or terms must be provided.

        Parameters
        ----------
            n_qubits: total number of qubits in the circuit

            costs: array containing values of the cost function at each computational basis state

            terms: list of weighted terms, where `term = (weight, [qubit indices])`
                The cost function for bitstring :math:`x` with bits :math:`x_j`
                taking values -1 or 1 (For example, `3 -> -1,1,1`) is given by
                a sum over terms of product of bitstring values at qubit indices:
                .. math::
                    C(x) = \\sum_{\text{terms}} w_i \\prod_{j \\in \\text{term}} x_j

        """
        self.n_qubits = n_qubits
        self.n_states = 2**n_qubits
        if costs is None:
            if terms is None:
                raise ValueError("Either costs or terms must be provided")
            self._hc_diag = self._diag_from_terms(terms)
        else:
            assert len(costs) == self.n_states
            self._hc_diag = self._diag_from_costs(costs)

    # -- Internal methods

    @abstractmethod
    def _diag_from_terms(self, terms: TermsType) -> typing.Any:
        """
        Precompute the diagonal of the cost Hamiltonian.
        Returns implementation-specific data type. For example,
        GPU simulator may return a GPU pointer. Consult the simulator
        implementation for details.

        Parameters
        ----------
            terms: list of tuples (coeff, [qubit indices])

        """
        ...

    @abstractmethod
    def _diag_from_costs(self, costs: CostsType) -> typing.Any:
        """
        Adapt the costs array to the simulator-specific datatype

        Parameters
        ----------
            costs: A sequence or a numpy array of length 2**n_qubits
        """
        ...

    # -- Public methods

    @abstractmethod
    def get_cost_diagonal(self) -> np.ndarray:
        """
        Return the diagonal of the cost Hamiltonian.
        Returns a numpy array.
        """
        ...

    @abstractmethod
    def simulate_qaoa(
        self,
        gammas: ParamType,
        betas: ParamType,
        sv0: np.ndarray | None = None,
        **kwargs,
    ) -> typing.Any:
        """
        simulator QAOA circuit
        Parameters
        ----------
            gammas: parameters for the phase separating layers
            betas: parameters for the mixing layers
            sv0: (optional) initial statevector, default is uniform superposition state
            kwargs: additional arguments for the simulator depending on the implementation

        Returns
        -------
            statevector: Depends on implementation

        """
        ...

    # -- Output methods

    @abstractmethod
    def get_expectation(self, result, costs: typing.Any = None, **kwargs) -> float:
        """
        Return the expectation value of the cost Hamiltonian

        Parameters
        ----------
            result: obtained from `sim.simulate_qaoa`
            costs: (optional) array containing values of the cost function at
                each computational basis state. Accepted types depend on the implementation.
        """
        ...

    @abstractmethod
    def get_overlap(self, result, costs: CostsType | None = None, indices: np.ndarray | Sequence[int] | None = None, **kwargs) -> float:
        """
        Return the overlap between the lowest energy state and the statevector
        Parameters
        ----------
            result: obtained from `sim.simulate_qaoa`
            costs: (optional) array containing values of the cost function at
                each computational basis state. Accepted types depend on the implementation
            **kwargs : additional arguments for the simulator depending on the implementation
        """
        ...

    @abstractmethod
    def get_statevector(self, result: typing.Any, **kwargs) -> np.ndarray:
        """
        Return the statevector as numpy array, which requires enough memory to
        store `2**n_qubits` complex numbers

        Parameters
        ----------
            result: obtained from `sim.simulate_qaoa`
        """
        ...

    @abstractmethod
    def get_probabilities(self, result: typing.Any, **kwargs) -> np.ndarray:
        """
        Return the probabilities of each computational basis state

        Parameters
        ----------
            result: obtained from `sim.simulate_qaoa`
        """
        ...
