import typing
import numpy as np
from ..qaoa_simulator_base import QAOAFastSimulatorBase
from .qaoa_fur import apply_qaoa_furx, apply_qaoa_furxy_complete, apply_qaoa_furxy_ring


class QAOAFastSimulatorPythonBase(QAOAFastSimulatorBase):
    @property
    def default_sv0(self):
        return np.full(self.n_states, 1.0 / np.sqrt(self.n_states), dtype="complex")

    def _apply_qaoa(self, sv: np.ndarray, gammas: typing.Sequence[float], betas: typing.Sequence[float], **kwargs):
        raise NotImplementedError

    def simulate_qaoa(
        self,
        gammas: typing.Sequence[float],
        betas: typing.Sequence[float],
        sv0: typing.Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        simulator QAOA circuit using FUR
        @param gammas parameters for the phase separating layers
        @param betas parameters for the mixing layers
        @param sv0 (optional) initial statevector, default is uniform superposition state
        @return statevector or vector of probabilities
        """
        sv = sv0.astype("complex") if sv0 is not None else self.default_sv0
        self._apply_qaoa(sv, gammas, betas, **kwargs)
        return sv


class QAOAFURXSimulator(QAOAFastSimulatorPythonBase):
    def _apply_qaoa(self, sv: np.ndarray, gammas: typing.Sequence[float], betas: typing.Sequence[float]):
        apply_qaoa_furx(sv, gammas, betas, self.hc_diag, self.n_qubits)


class QAOAFURXYRingSimulator(QAOAFastSimulatorPythonBase):
    def _apply_qaoa(self, sv: np.ndarray, gammas: typing.Sequence[float], betas: typing.Sequence[float], n_trotters: int = 1):
        apply_qaoa_furxy_ring(sv, gammas, betas, self.hc_diag, self.n_qubits, n_trotters=n_trotters)


class QAOAFURXYCompleteSimulator(QAOAFastSimulatorPythonBase):
    def _apply_qaoa(self, sv: np.ndarray, gammas: typing.Sequence[float], betas: typing.Sequence[float], n_trotters: int = 1):
        apply_qaoa_furxy_complete(sv, gammas, betas, self.hc_diag, self.n_qubits, n_trotters=n_trotters)
