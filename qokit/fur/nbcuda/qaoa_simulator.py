###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
from __future__ import annotations
from collections.abc import Sequence
import numpy as np
import numba.cuda
import warnings

from ..qaoa_simulator_base import QAOAFastSimulatorBase, ParamType, CostsType, TermsType
from .qaoa_fur import apply_qaoa_furx, apply_qaoa_furxy_complete, apply_qaoa_furxy_ring
from ..diagonal_precomputation import precompute_gpu
from .diagonal import apply_diagonal_from_terms, terms_to_device_arrays
from .utils import norm_squared, initialize_uniform, multiply, sum_reduce, copy

# Threshold (qubits) above which the lazy diagonal mode is auto-selected.
#
# At n >= 22 the precomputed diagonal array is >= 16 MB of GPU memory.  The
# lazy path saves this allocation (enabling n=26+ simulation without OOM) at
# the cost of additional arithmetic per QAOA step.  Profiled on RTX 3070Ti
# Laptop GPU for 100-edge MAXCUT problems with p=3:
#
#   n=20  lazy 1.5x *faster* end-to-end (avoids CPU precompute + PCIe xfer)
#   n=22  break-even (precomputed 18.8 ms, lazy 28.5 ms)
#   n=26  precomputed 2x faster per step, but needs 256 MB GPU memory
#
# Callers can pass ``lazy_diagonal=True/False`` to override the auto-selection.
_LAZY_DIAGONAL_THRESHOLD_QUBITS = 22

DeviceArray = numba.cuda.devicearray.DeviceNDArray


class QAOAFastSimulatorGPUBase(QAOAFastSimulatorBase):
    """GPU-accelerated QAOA simulator base.

    When constructed from *terms* and ``n_qubits >= _LAZY_DIAGONAL_THRESHOLD_QUBITS``,
    the diagonal of the cost Hamiltonian is **not** precomputed into a 2^n
    array.  Instead, each call to ``apply_diagonal`` recomputes the Ising
    energy on-the-fly inside the CUDA kernel (Issue #35).  This avoids
    allocating up to 1 GB of GPU memory for n=28 qubits and eliminates
    the CPU→GPU transfer that dominates wall-clock time at large n.
    """

    def __init__(
        self,
        n_qubits: int,
        costs: CostsType | None = None,
        terms: TermsType | None = None,
        lazy_diagonal: bool | None = None,
    ) -> None:
        # Store terms before super().__init__ because _diag_from_terms may set
        # _use_lazy and _terms_device based on them.
        self._use_lazy: bool = False
        self._terms_device_coef = None
        self._terms_device_mask = None

        if lazy_diagonal is None:
            lazy_diagonal = (terms is not None) and (n_qubits >= _LAZY_DIAGONAL_THRESHOLD_QUBITS)

        if lazy_diagonal and terms is not None:
            self._use_lazy = True
            self._terms_device_coef, self._terms_device_mask = terms_to_device_arrays(terms)
            # Skip precomputation — base class __init__ will call _diag_from_terms,
            # which we override to return a sentinel None when lazy mode is active.

        super().__init__(n_qubits, costs, terms)
        self._sv_device = numba.cuda.device_array(self.n_states, dtype="complex")  # type: ignore

    def _diag_from_costs(self, costs: CostsType) -> DeviceArray:
        return numba.cuda.to_device(costs)

    def _diag_from_terms(self, terms: TermsType, rank: int = 0) -> DeviceArray | None:
        if self._use_lazy:
            # Lazy mode: no precomputed diagonal array.  Return None as sentinel;
            # _apply_diagonal_phase handles both paths.
            return None
        out = numba.cuda.device_array(self.n_states, dtype="float32")  # type: ignore
        precompute_gpu(rank, self.n_qubits, terms, out)
        return out

    def _apply_diagonal_phase(self, sv: DeviceArray, gamma: float, offset: int = 0) -> None:
        """Apply e^{-i gamma H_C / 2} to *sv* in-place.

        Uses on-the-fly energy computation (lazy mode) or a precomputed
        diagonal lookup depending on how the simulator was constructed.
        """
        if self._use_lazy:
            apply_diagonal_from_terms(sv, gamma, self._terms_device_coef, self._terms_device_mask, offset)
        else:
            from .diagonal import apply_diagonal

            apply_diagonal(sv, gamma, self._hc_diag)

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
        if self._hc_diag is None:
            raise RuntimeError(
                "get_cost_diagonal() is not available in lazy diagonal mode "
                "(n_qubits >= _LAZY_DIAGONAL_THRESHOLD_QUBITS and terms provided). "
                "Pass costs= explicitly or construct with lazy_diagonal=False."
            )
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

    def get_expectation(self, result: DeviceArray, costs: DeviceArray | np.ndarray | None = None, optimization_type="min", **kwargs) -> float:
        if costs is None:
            if self._hc_diag is None:
                raise RuntimeError(
                    "get_expectation() requires costs= in lazy diagonal mode. " "Pass the cost diagonal explicitly or construct with lazy_diagonal=False."
                )
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
        if optimization_type == "max":
            return -1 * sum_reduce(result).real  # type: ignore
        else:
            return sum_reduce(result).real

    def get_overlap(
        self, result: DeviceArray, costs: CostsType | None = None, indices: np.ndarray | Sequence[int] | None = None, optimization_type="min", **kwargs
    ) -> float:
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
            if optimization_type == "max":
                val = costs_t.max()
            else:
                val = costs_t.min()
            indices_sel = costs_t == val
        else:
            indices_sel = indices
        return probs[indices_sel].sum().item()


class QAOAFURXSimulatorGPU(QAOAFastSimulatorGPUBase):
    def _apply_qaoa(self, gammas: Sequence[float], betas: Sequence[float], **kwargs):
        apply_qaoa_furx(
            self._sv_device,
            gammas,
            betas,
            self._hc_diag,
            self.n_qubits,
            apply_diag_fn=self._apply_diagonal_phase if self._use_lazy else None,
        )


class QAOAFURXYRingSimulatorGPU(QAOAFastSimulatorGPUBase):
    def _apply_qaoa(self, gammas: Sequence[float], betas: Sequence[float], **kwargs):
        n_trotters = kwargs.get("n_trotters", 1)
        apply_qaoa_furxy_ring(
            self._sv_device,
            gammas,
            betas,
            self._hc_diag,
            self.n_qubits,
            n_trotters=n_trotters,
            apply_diag_fn=self._apply_diagonal_phase if self._use_lazy else None,
        )


class QAOAFURXYCompleteSimulatorGPU(QAOAFastSimulatorGPUBase):
    def _apply_qaoa(self, gammas: Sequence[float], betas: Sequence[float], **kwargs):
        n_trotters = kwargs.get("n_trotters", 1)
        apply_qaoa_furxy_complete(
            self._sv_device,
            gammas,
            betas,
            self._hc_diag,
            self.n_qubits,
            n_trotters=n_trotters,
            apply_diag_fn=self._apply_diagonal_phase if self._use_lazy else None,
        )
