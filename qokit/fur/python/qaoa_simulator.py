###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
from __future__ import annotations

from collections.abc import Sequence
import numpy as np

from ..qaoa_simulator_base import (
    QAOAFastSimulatorBase,
    CostsType,
    TermsType,
    ParamType,
)
from ..diagonal_precomputation import precompute_vectorized_cpu_parallel
from .qaoa_fur import (
    apply_qaoa_furx,
    apply_qaoa_furxy_complete,
    apply_qaoa_furxy_ring,
)

# ★ NEW – fixed-point helpers (only used when caller sets ``quant_bits``)
from .quant_utils  import quantise_fp, dequantise_fp, _BLOCK_DEF
# ──────────────────────────────────────────────────────────────────────────────
def little_to_big_endian(arr: np.ndarray) -> np.ndarray:
    n = int(np.log2(len(arr)))
    bin_idx = np.vectorize(lambda x: np.binary_repr(x, width=n))(np.arange(len(arr)))
    rev_idx = np.array([int(b[::-1], 2) for b in bin_idx])
    return arr[rev_idx]


# ──────────────────────────────────────────────────────────────────────────────
class QAOAFastSimulatorPythonBase(QAOAFastSimulatorBase):
    _hc_diag: np.ndarray

    # NEW – allow user-selected dtype for the default |+…+⟩ state
    def __init__(
        self,
        *args,
        sv_dtype: np.dtype | str = "complex128",   # default = old behaviour
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._sv_dtype = np.dtype(sv_dtype)

    # ---------------------------------------------------------------------
    def _diag_from_costs(self, costs: CostsType):
        return np.asarray(costs, dtype="float")

    def _diag_from_terms(self, terms: TermsType):
        return precompute_vectorized_cpu_parallel(terms, 0.0, self.n_qubits)

    def get_cost_diagonal(self) -> np.ndarray:
        return self._hc_diag

    # EDIT – uses the configurable dtype
    @property
    def default_sv0(self):
        return np.full(self.n_states, 1.0 / np.sqrt(self.n_states), dtype=self._sv_dtype)

    # ---------------------------------------------------------------------
    def _apply_qaoa(
        self,
        sv: np.ndarray,
        gammas: Sequence[float],
        betas: Sequence[float],
        **kwargs,
    ):
        raise NotImplementedError

    # ---------------------------------------------------------------------
    # ------------------------------------------------------------------
    def simulate_qaoa(
        self,
        gammas: ParamType,
        betas:  ParamType,
        sv0:    np.ndarray | None = None,
        *,
        quant_bits: int | None = None,
        block_size: int        = _BLOCK_DEF,
        **kwargs,
    ) -> np.ndarray:
        """
        Run FUR-based QAOA with optional fixed-point round-trip.

        Extra keywords
        --------------
        init_dtype  : np.dtype | str – one-off precision override
        quant_bits  : 8 / 16 …       – enable FP quantisation
        block_size  : int            – per-block length for FP scheme
        """

        # -- (0) optional one-shot precision override ----------------------
        if "init_dtype" in kwargs:
            self._sv_dtype = np.dtype(kwargs.pop("init_dtype"))

        # -- (1) build / cast the initial statevector ----------------------
        sv = (
            sv0.astype(self._sv_dtype, copy=False)
            if sv0 is not None else
            self.default_sv0
        )

        # -- (2) optional fixed-point quantisation round-trip --------------
        if quant_bits is not None:
            rq, iq, scl = quantise_fp(
                sv, bits=quant_bits, block_size=block_size
            )
            sv = dequantise_fp(
                rq, iq, scl, bits=quant_bits, block_size=block_size
            )

        # -- (3) run the circuit in-place ----------------------------------
        self._apply_qaoa(sv, list(gammas), list(betas), **kwargs)
        return sv

    # ------------------------------------------------------------------


    # ------------------------------------------------------------------ outputs
    def get_statevector(self, result: np.ndarray, **kwargs) -> np.ndarray:
        return result

    def get_probabilities(self, result: np.ndarray, **kwargs) -> np.ndarray:
        return np.abs(result) ** 2

    def get_expectation(
        self,
        result: np.ndarray,
        costs: np.ndarray | None = None,
        optimization_type: str = "min",
        **kwargs,
    ) -> float:
        if costs is None:
            costs = self._hc_diag
        val = np.dot(costs, np.abs(result) ** 2)
        return -val if optimization_type == "max" else val

    def get_overlap(
        self,
        result: np.ndarray,
        costs: CostsType | None = None,
        indices: np.ndarray | Sequence[int] | None = None,
        optimization_type: str = "min",
        **kwargs,
    ) -> float:
        probs = self.get_probabilities(result)
        if indices is None:
            costs = self._hc_diag if costs is None else self._diag_from_costs(costs)
            target = costs.max() if optimization_type == "max" else costs.min()
            indices = (costs == target).nonzero()
        return probs[indices].sum()


# ────────────────────────── concrete simulators ─────────────────────────────
class QAOAFURXSimulator(QAOAFastSimulatorPythonBase):
    def _apply_qaoa(
        self, sv: np.ndarray, gammas: Sequence[float], betas: Sequence[float], **kwargs
    ):
        apply_qaoa_furx(sv, gammas, betas, self._hc_diag, self.n_qubits)


class QAOAFURXYRingSimulator(QAOAFastSimulatorPythonBase):
    def _apply_qaoa(
        self, sv: np.ndarray, gammas: Sequence[float], betas: Sequence[float], **kwargs
    ):
        n_trotters = kwargs.get("n_trotters", 1)
        apply_qaoa_furxy_ring(
            sv, gammas, betas, self._hc_diag, self.n_qubits, n_trotters=n_trotters
        )


class QAOAFURXYCompleteSimulator(QAOAFastSimulatorPythonBase):
    def _apply_qaoa(
        self, sv: np.ndarray, gammas: Sequence[float], betas: Sequence[float], **kwargs
    ):
        n_trotters = kwargs.get("n_trotters", 1)
        apply_qaoa_furxy_complete(
            sv, gammas, betas, self._hc_diag, self.n_qubits, n_trotters=n_trotters
        )
