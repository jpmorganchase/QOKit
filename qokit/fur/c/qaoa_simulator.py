###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
from __future__ import annotations
from collections.abc import Sequence
import numpy as np

from ..qaoa_simulator_base import QAOAFastSimulatorBase, CostsType, TermsType, ParamType
from ..diagonal_precomputation import precompute_vectorized_cpu_parallel

from .gates import ComplexArray
from . import csim


class QAOAFastSimulatorCBase(QAOAFastSimulatorBase):
    _hc_diag: np.ndarray

    def _diag_from_costs(self, costs: CostsType) -> np.ndarray:
        return np.asarray(costs, dtype="float")

    def _diag_from_terms(self, terms: TermsType):
        return precompute_vectorized_cpu_parallel(terms, 0.0, self.n_qubits)

    def get_cost_diagonal(self) -> np.ndarray:
        return self._hc_diag

    @property
    def default_sv0(self):
        return ComplexArray(
            np.full(self.n_states, 1.0 / np.sqrt(self.n_states), dtype="float"),
            np.zeros(self.n_states, dtype="float"),
        )

    def warmstart_sv0(self, init_rots):
        state = 1.0

        for init_rot in init_rots:
            qubit_state = np.array(
                [
                    np.cos(init_rot / 2),
                    np.sin(init_rot / 2),
                    # np.exp(-1j * np.pi/2) * np.sin(init_rot/2)
                ]
            )
            # state = np.kron(state, qubit_state) #original as GW
            state = np.kron(qubit_state, state)

        return ComplexArray(state.real.astype("float"), state.imag.astype("float"))

    def _apply_qaoa(self, sv: ComplexArray, gammas: Sequence[float], betas: Sequence[float], **kwargs):
        raise NotImplementedError

    def simulate_ws_qaoa(
        self,
        gammas: ParamType,
        betas: ParamType,
        init_rots: ParamType,
        sv0: np.ndarray | None = None,
        **kwargs,
    ) -> ComplexArray:
        """
        simulator QAOA circuit using FUR
        @param gammas parameters for the phase separating layers
        @param betas parameters for the mixing layers
        @param sv0 (optional) initial statevector, default is uniform superposition state
        @return statevector or vector of probabilities
        """
        sv = ComplexArray(sv0.real.astype("float"), sv0.imag.astype("float")) if sv0 is not None else self.warmstart_sv0(init_rots)
        self._apply_qaoa(sv, list(gammas), list(betas), init_rots, **kwargs)
        return sv

    def simulate_qaoa(
        self,
        gammas: ParamType,
        betas: ParamType,
        sv0: np.ndarray | None = None,
        **kwargs,
    ) -> ComplexArray:
        sv = ComplexArray(sv0.real.astype("float"), sv0.imag.astype("float")) if sv0 is not None else self.default_sv0
        self._apply_qaoa(sv, list(gammas), list(betas), **kwargs)
        return sv

    def get_statevector(self, result: ComplexArray, **kwargs) -> np.ndarray:
        return result.get_complex()

    def get_probabilities(self, result: ComplexArray, **kwargs) -> np.ndarray:
        return result.get_norm_squared()

    def get_expectation(self, result: ComplexArray, costs: np.ndarray | None = None, optimization_type="min", **kwargs) -> float:
        if costs is None:
            costs = self._hc_diag
        if optimization_type == "max":
            costs = -1 * np.asarray(costs)
        return np.dot(costs, self.get_probabilities(result, **kwargs))

    def get_std(self, result: ComplexArray, costs: np.ndarray | None = None, **kwargs) -> float:
        if costs is None:
            costs = self._hc_diag
        probs = self.get_probabilities(result)
        return np.sqrt(max(np.dot(costs**2, probs) - np.dot(costs, probs) ** 2, 0))

    def get_overlap(
        self, result: ComplexArray, costs: CostsType | None = None, indices: np.ndarray | Sequence[int] | None = None, optimization_type="min", **kwargs
    ) -> float:
        """
        Compute the overlap between the statevector and the ground state

        Parameters
        ----------
            result: statevector
            costs: (optional) diagonal of the cost Hamiltonian
            indices: (optional) indices of the ground state in the statevector
            preserve_state: (optional) if True, allocate a new array for probabilities
        """
        probs = self.get_probabilities(result, **kwargs)
        if indices is None:
            if costs is None:
                costs = self._hc_diag
            else:
                costs = self._diag_from_costs(costs)
            if optimization_type == "max":
                costs = -1 * np.asarray(costs)
            minval = costs.min()
            indices = (costs == minval).nonzero()
        return probs[indices].sum()


class QAOAFURXSimulatorC(QAOAFastSimulatorCBase):
    def _apply_qaoa(self, sv: ComplexArray, gammas: Sequence[float], betas: Sequence[float], **kwargs):
        csim.apply_qaoa_furx(
            sv.real,
            sv.imag,
            gammas,
            betas,
            self._hc_diag,
            self.n_qubits,
        )


class QAOAFURXZSimulatorC(QAOAFastSimulatorCBase):
    def _apply_qaoa(self, sv: ComplexArray, gammas: Sequence[float], betas: Sequence[float], init_rots: Sequence[float], **kwargs):
        csim.apply_qaoa_furxz(
            sv.real,
            sv.imag,
            gammas,
            betas,
            init_rots,
            self._hc_diag,
            self.n_qubits,
        )


class QAOAFURXYRingSimulatorC(QAOAFastSimulatorCBase):
    def _apply_qaoa(self, sv: ComplexArray, gammas: Sequence[float], betas: Sequence[float], **kwargs):
        n_trotters = kwargs.get("n_trotters", 1)
        csim.apply_qaoa_furxy_ring(
            sv.real,
            sv.imag,
            gammas,
            betas,
            self._hc_diag,
            self.n_qubits,
            n_trotters,
        )


class QAOAFURXYCompleteSimulatorC(QAOAFastSimulatorCBase):
    def _apply_qaoa(self, sv: ComplexArray, gammas: Sequence[float], betas: Sequence[float], **kwargs):
        n_trotters = kwargs.get("n_trotters", 1)
        csim.apply_qaoa_furxy_complete(
            sv.real,
            sv.imag,
            gammas,
            betas,
            self._hc_diag,
            self.n_qubits,
            n_trotters,
        )
