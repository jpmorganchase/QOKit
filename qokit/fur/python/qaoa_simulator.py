# python/qaoa_simulator.py
# ──────────────────────────────────────────────────────────────────────────────
# SPDX-License-Identifier: Apache-2.0
# Copyright JP Morgan Chase & Co
# ──────────────────────────────────────────────────────────────────────────────

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
    apply_qaoa_furxy_ring,
    apply_qaoa_furxy_complete,
)
from .quant_utils import quantise_fp, dequantise_fp, _BLOCK_DEF

# ──────────────────────────────────────────────────────────────────────────────

class QAOAFastSimulatorPythonBase(QAOAFastSimulatorBase):
    def __init__(
        self,
        n_qubits: int,
        costs: CostsType | None = None,
        terms: TermsType | None = None,
        sv_dtype: np.dtype | str = "complex128",
    ) -> None:
        super().__init__(n_qubits, costs=costs, terms=terms)
        self._sv_dtype = np.dtype(sv_dtype)

    def _diag_from_terms(self, terms: TermsType) -> np.ndarray:
        return precompute_vectorized_cpu_parallel(terms, 0.0, self.n_qubits)

    def _diag_from_costs(self, costs: CostsType) -> np.ndarray:
        return np.asarray(costs, dtype="float")

    def get_cost_diagonal(self) -> np.ndarray:
        return self._hc_diag

    @property
    def default_sv0(self) -> np.ndarray:
        return np.full(self.n_states,
                       1/np.sqrt(self.n_states),
                       dtype=self._sv_dtype)

    def simulate_qaoa(
        self,
        gammas: ParamType,
        betas: ParamType,
        sv0: np.ndarray | None = None,
        *,
        quant_bits: int | None = None,
        block_size: int = _BLOCK_DEF,
        init_dtype: np.dtype | str | None = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Run FUR-based QAOA with per-layer fixed-point quantization.

        quant_bits: if set, bit-width of fixed-point quantization
        block_size: block size for per-block scaling
        init_dtype: optional override of initial state dtype
        """

        # Override initial-state dtype?
        if init_dtype is not None:
            self._sv_dtype = np.dtype(init_dtype)

        # Build initial state
        if sv0 is not None:
            sv = sv0.astype(self._sv_dtype, copy=False)
        else:
            sv = self.default_sv0.copy()

        # Initial quant-roundtrip?
        if quant_bits is not None:
            rq, iq, scales = quantise_fp(
                sv, bits=quant_bits, block_size=block_size
            )
            sv = dequantise_fp(
                rq, iq, scales,
                bits=quant_bits, block_size=block_size,
                renorm=False
            )

        # Per-layer cost → quant → mixer → quant
        for gamma, beta in zip(gammas, betas):
            # Phase separator
            sv *= np.exp(-0.5j * gamma * self._hc_diag)
            if quant_bits is not None:
                rq, iq, scales = quantise_fp(
                    sv, bits=quant_bits, block_size=block_size
                )
                sv = dequantise_fp(
                    rq, iq, scales,
                    bits=quant_bits, block_size=block_size,
                    renorm=False
                )

            # Mixer step
            self._apply_mixer_layer(sv, beta, **kwargs)
            if quant_bits is not None:
                rq, iq, scales = quantise_fp(
                    sv, bits=quant_bits, block_size=block_size
                )
                sv = dequantise_fp(
                    rq, iq, scales,
                    bits=quant_bits, block_size=block_size,
                    renorm=False
                )

        return sv

    def _apply_mixer_layer(self, sv: np.ndarray, beta: float, **kwargs) -> None:
        """Dispatch a single mixer step in-place."""
        if isinstance(self, QAOAFURXSimulator):
            apply_qaoa_furx(sv, [0.0], [beta], self._hc_diag, self.n_qubits)
        elif isinstance(self, QAOAFURXYRingSimulator):
            apply_qaoa_furxy_ring(
                sv, [0.0], [beta], self._hc_diag, self.n_qubits,
                n_trotters=kwargs.get("n_trotters", 1)
            )
        elif isinstance(self, QAOAFURXYCompleteSimulator):
            apply_qaoa_furxy_complete(
                sv, [0.0], [beta], self._hc_diag, self.n_qubits,
                n_trotters=kwargs.get("n_trotters", 1)
            )
        else:
            raise RuntimeError("Unknown Python FUR mixer type")

    # Concrete implementations of abstract output methods:

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
        val = float(np.dot(costs, np.abs(result) ** 2))
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
            costs_arr = self._hc_diag if costs is None else self._diag_from_costs(costs)
            target = costs_arr.max() if optimization_type == "max" else costs_arr.min()
            indices = (costs_arr == target).nonzero()
        return float(probs[indices].sum())


# ──────────────────────────────────────────────────────────────────────────────

class QAOAFURXSimulator(QAOAFastSimulatorPythonBase):
    def _apply_qaoa(
        self,
        sv: np.ndarray,
        gammas: Sequence[float],
        betas: Sequence[float],
        **kwargs,
    ):
        apply_qaoa_furx(sv, gammas, betas, self._hc_diag, self.n_qubits)


class QAOAFURXYRingSimulator(QAOAFastSimulatorPythonBase):
    def _apply_qaoa(
        self,
        sv: np.ndarray,
        gammas: Sequence[float],
        betas: Sequence[float],
        **kwargs,
    ):
        apply_qaoa_furxy_ring(
            sv, gammas, betas, self._hc_diag, self.n_qubits,
            n_trotters=kwargs.get("n_trotters", 1)
        )


class QAOAFURXYCompleteSimulator(QAOAFastSimulatorPythonBase):
    def _apply_qaoa(
        self,
        sv: np.ndarray,
        gammas: Sequence[float],
        betas: Sequence[float],
        **kwargs,
    ):
        apply_qaoa_furxy_complete(
            sv, gammas, betas, self._hc_diag, self.n_qubits,
            n_trotters=kwargs.get("n_trotters", 1)
        )
