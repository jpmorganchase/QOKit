###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
from __future__ import annotations
from collections.abc import Sequence
import numpy as np
import numba.cuda
import warnings

from qokit.fur.qaoa_simulator_base import TermsType

from ..qaoa_simulator_base import QAOAFastSimulatorBase, ParamType, CostsType, TermsType
from .qaoa_fur import apply_qaoa_furx, apply_qaoa_furxy_complete, apply_qaoa_furxy_ring
from ..diagonal_precomputation import precompute_gpu
from .utils import norm_squared, initialize_uniform, multiply, sum_reduce, copy


DeviceArray = numba.cuda.devicearray.DeviceNDArray


class QAOAFastSimulatorGPUBase(QAOAFastSimulatorBase):
    def __init__(self, n_qubits: int, costs: CostsType | None = None, terms: TermsType | None = None) -> None:
        super().__init__(n_qubits, costs, terms)
        self._sv_device = numba.cuda.device_array(self.n_states, dtype="complex")  # type: ignore

    def _diag_from_costs(self, costs: CostsType) -> DeviceArray:
        return numba.cuda.to_device(costs)

    def _diag_from_terms(self, terms: TermsType) -> DeviceArray:
        out = numba.cuda.device_array(self.n_states, dtype="float32")  # type: ignore
        precompute_gpu(0, self.n_qubits, terms, out)
        return out

    def _apply_qaoa(self, gammas: Sequence[float], betas: Sequence[float], **kwargs):
        raise NotImplementedError

    def _initialize(
        self,
        sv0: np.ndarray | None = None,
    ) -> None:
        if sv0 is None:
            initialize_uniform(self._sv_device)
        else:
            numba.cuda.to_device(np.asarray(sv0, dtype="complex"), to=self._sv_device)

    def get_cost_diagonal(self) -> np.ndarray:
        return self._hc_diag.copy_to_host()

    def simulate_qaoa(
        self,
        gammas: ParamType,
        betas: ParamType,
        sv0: np.ndarray | None = None,
        **kwargs,
    ) -> DeviceArray:
        """
        See QAOAFastSimulatorBase.simulate_qaoa
        """
        self._initialize(sv0=sv0)
        self._apply_qaoa(list(gammas), list(betas), **kwargs)
        return self._sv_device

    def get_statevector(self, result: DeviceArray, **kwargs) -> np.ndarray:
        return result.copy_to_host()

    def get_probabilities(self, result: DeviceArray, **kwargs) -> np.ndarray:
        preserve_state = kwargs.get("preserve_state", True)
        if preserve_state:
            result_orig = result
            result = numba.cuda.device_array_like(result_orig)
            copy(result, result_orig)
        norm_squared(result)
        return result.copy_to_host().real

    def get_expectation(self, result: DeviceArray, costs: DeviceArray | np.ndarray | None = None, **kwargs) -> float:
        if costs is None:
            costs = self._hc_diag
        else:
            costs = self._diag_from_costs(costs)
        preserve_state = kwargs.get("preserve_state", True)
        if preserve_state:
            result_orig = result
            result = numba.cuda.device_array_like(result_orig)
            copy(result, result_orig)
        norm_squared(result)
        multiply(result, costs)
        return sum_reduce(result).real  # type: ignore

    def get_overlap(self, result: DeviceArray, costs: CostsType | None = None, indices: np.ndarray | Sequence[int] | None = None, **kwargs) -> float:
        """
        Compute the overlap between the statevector and the ground state

        Requires cupy to be installed

        Parameters
        ----------
            result: statevector
            costs: (optional) diagonal of the cost Hamiltonian
            indices: (optional) indices of the ground state in the statevector
            preserve_state: (optional) if True, allocate a new array for probabilities
        """
        try:
            import cupy as cp
        except ImportError:
            warnings.warn("Cupy import failed, which may cause a performance drop for overlap calculation.", RuntimeWarning)
            import numpy as cp

        probs = self.get_probabilities(result, **kwargs)
        probs: cp.ndarray = cp.asarray(probs)
        if indices is None:
            if costs is None:
                costs_t = self._hc_diag
            else:
                costs_t = self._diag_from_costs(costs)

            # pass without copy
            costs_t: cp.ndarray = cp.asarray(costs_t)
            minval = costs_t.min()
            indices_sel = costs_t == minval
        else:
            indices_sel = indices
        return probs[indices_sel].sum()


class QAOAFURXSimulatorGPU(QAOAFastSimulatorGPUBase):
    def _apply_qaoa(self, gammas: Sequence[float], betas: Sequence[float], **kwargs):
        apply_qaoa_furx(self._sv_device, gammas, betas, self._hc_diag, self.n_qubits)


class QAOAFURXYRingSimulatorGPU(QAOAFastSimulatorGPUBase):
    def _apply_qaoa(self, gammas: Sequence[float], betas: Sequence[float], **kwargs):
        n_trotters = kwargs.get("n_trotters", 1)
        apply_qaoa_furxy_ring(self._sv_device, gammas, betas, self._hc_diag, self.n_qubits, n_trotters=n_trotters)


class QAOAFURXYCompleteSimulatorGPU(QAOAFastSimulatorGPUBase):
    def _apply_qaoa(self, gammas: Sequence[float], betas: Sequence[float], **kwargs):
        n_trotters = kwargs.get("n_trotters", 1)
        apply_qaoa_furxy_complete(self._sv_device, gammas, betas, self._hc_diag, self.n_qubits, n_trotters=n_trotters)
