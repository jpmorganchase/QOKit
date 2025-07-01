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

# fixed-point helpers (Python-only)
from .quant_utils import quantise_fp, dequantise_fp, _BLOCK_DEF


class QAOAFastSimulatorCBase(QAOAFastSimulatorBase):
    _hc_diag: np.ndarray

    def __init__(self, *args, **kwargs):
        # strip Python-only kwargs so they don't reach the Base
        for _k in ("sv_dtype", "quant_bits", "block_size", "renorm"):
            kwargs.pop(_k, None)
        super().__init__(*args, **kwargs)

    def _diag_from_costs(self, costs: CostsType) -> np.ndarray:
        return np.asarray(costs, dtype="float")

    def _diag_from_terms(self, terms: TermsType) -> np.ndarray:
        return precompute_vectorized_cpu_parallel(terms, 0.0, self.n_qubits)

    def get_cost_diagonal(self) -> np.ndarray:
        return self._hc_diag

    @property
    def default_sv0(self) -> ComplexArray:
        # uniform |+...+> in double-precision real+imag
        real = np.full(self.n_states, 1.0/np.sqrt(self.n_states), dtype="float")
        imag = np.zeros(self.n_states, dtype="float")
        return ComplexArray(real, imag)

    def _apply_qaoa(
        self,
        sv: ComplexArray,
        gammas: Sequence[float],
        betas: Sequence[float],
        **kwargs,
    ):
        raise NotImplementedError


    def simulate_qaoa(
        self,
        gammas: ParamType,
        betas: ParamType,
        sv0: np.ndarray | None = None,
        *,
        sv_dtype: str | np.dtype = "complex128",
        quant_bits: int | None    = None,
        block_size: int           = _BLOCK_DEF,
        renorm: bool              = False,
        **kwargs,
    ) -> ComplexArray:
        """
        Same signature as the Python FUR simulator.

        Extra kwargs
        ------------
        sv_dtype   : 'complex64' or 'complex128' for the working FP copy
        quant_bits : if int → fixed-point round-trip on the *input* statevec
        block_size : length of each block for per-block scaling
        renorm     : whether to re-normalize after dequantisation
        """
        work_dtype = np.dtype(sv_dtype)

        # 1) build / cast initial state
        if sv0 is None:
            # uniform superposition in desired precision
            base = np.full(self.n_states, 1.0/np.sqrt(self.n_states), dtype=work_dtype)
            real = base.real.astype("float")
            imag = base.imag.astype("float")
            sv   = ComplexArray(real, imag)
        else:
            # user-supplied statevector
            real = sv0.real.astype(work_dtype).real.astype("float")
            imag = sv0.imag.astype(work_dtype).real.astype("float")
            sv   = ComplexArray(real, imag)

        # 2) optional fixed-point round-trip on input
        if quant_bits is not None:
            full = sv.get_complex()
            rq, iq, scales = quantise_fp(full,
                                          bits=quant_bits,
                                          block_size=block_size)
            deq = dequantise_fp(rq,
                                iq,
                                scales,
                                bits=quant_bits,
                                block_size=block_size,
                                renorm=renorm)
            sv = ComplexArray(deq.real.astype("float"),
                              deq.imag.astype("float"))

        # 3) run the FUR circuit in-place
        self._apply_qaoa(sv, list(gammas), list(betas), **kwargs)
        return sv


    def get_statevector(self, result: ComplexArray, **kwargs) -> np.ndarray:
        return result.get_complex()

    def get_probabilities(self, result: ComplexArray, **kwargs) -> np.ndarray:
        return result.get_norm_squared()

    def get_expectation(
        self,
        result: ComplexArray,
        costs: np.ndarray | None = None,
        optimization_type="min",
        **kwargs,
    ) -> float:
        if costs is None:
            costs = self._hc_diag
        if optimization_type == "max":
            costs = -1 * np.asarray(costs)
        return np.dot(costs, self.get_probabilities(result, **kwargs))

    def get_overlap(
        self,
        result: ComplexArray,
        costs: CostsType | None = None,
        indices: np.ndarray | Sequence[int] | None = None,
        optimization_type="min",
        **kwargs,
    ) -> float:
        probs = self.get_probabilities(result, **kwargs)
        if indices is None:
            if costs is None:
                costs = self._hc_diag
            else:
                costs = self._diag_from_costs(costs)
            if optimization_type == "max":
                costs = -1 * np.asarray(costs)
            target = costs.min()
            indices = (costs == target).nonzero()
        return probs[indices].sum()



class QAOAFURXSimulatorC(QAOAFastSimulatorCBase):
    def _apply_qaoa(
        self,
        sv: ComplexArray,
        gammas: Sequence[float],
        betas: Sequence[float],
        **kwargs,
    ):
        csim.apply_qaoa_furx(
            sv.real,
            sv.imag,
            gammas,
            betas,
            self._hc_diag,
            self.n_qubits,
        )


class QAOAFURXYRingSimulatorC(QAOAFastSimulatorCBase):
    def _apply_qaoa(
        self,
        sv: ComplexArray,
        gammas: Sequence[float],
        betas: Sequence[float],
        **kwargs,
    ):
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
    def _apply_qaoa(
        self,
        sv: ComplexArray,
        gammas: Sequence[float],
        betas: Sequence[float],
        **kwargs,
    ):
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
