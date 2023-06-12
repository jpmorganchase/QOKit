import typing
import numpy as np

from ..qaoa_simulator_base import QAOAFastSimulatorBase

from .gates import ComplexArray
from . import csim


class QAOAFastSimulatorCBase(QAOAFastSimulatorBase):
    def __init__(self, n_qubits: int, costs: typing.Sequence[float]) -> None:
        super().__init__(n_qubits, np.asarray(costs, dtype="float"))

    @property
    def default_sv0(self):
        return ComplexArray(
            np.full(self.n_states, 1.0 / np.sqrt(self.n_states), dtype="float"),
            np.zeros(self.n_states, dtype="float"),
        )

    def _apply_qaoa(self, sv: np.ndarray, gammas: typing.Sequence[float], betas: typing.Sequence[float], **kwargs):
        raise NotImplementedError

    def simulate_qaoa(
        self,
        gammas: typing.Sequence[float],
        betas: typing.Sequence[float],
        sv0: typing.Optional[np.ndarray] = None,
        **kwargs,
    ) -> ComplexArray:
        """
        simulator QAOA circuit using FUR
        @param gammas parameters for the phase separating layers
        @param betas parameters for the mixing layers
        @param sv0 (optional) initial statevector, default is uniform superposition state
        @return statevector or vector of probabilities
        """
        sv = ComplexArray(sv0.real.astype("float"), sv0.imag.astype("float")) if sv0 is not None else self.default_sv0
        self._apply_qaoa(sv, gammas, betas, **kwargs)
        return sv


class QAOAFURXSimulatorC(QAOAFastSimulatorCBase):
    def _apply_qaoa(self, sv: np.ndarray, gammas: typing.Sequence[float], betas: typing.Sequence[float]):
        csim.apply_qaoa_furx(
            sv.real,
            sv.imag,
            gammas,
            betas,
            self.hc_diag,
            self.n_qubits,
        )


class QAOAFURXYRingSimulatorC(QAOAFastSimulatorCBase):
    def _apply_qaoa(self, sv: np.ndarray, gammas: typing.Sequence[float], betas: typing.Sequence[float], n_trotters: int = 1):
        csim.apply_qaoa_furxy_ring(
            sv.real,
            sv.imag,
            gammas,
            betas,
            self.hc_diag,
            self.n_qubits,
            n_trotters,
        )


class QAOAFURXYCompleteSimulatorC(QAOAFastSimulatorCBase):
    def _apply_qaoa(self, sv: np.ndarray, gammas: typing.Sequence[float], betas: typing.Sequence[float], n_trotters: int = 1):
        csim.apply_qaoa_furxy_complete(
            sv.real,
            sv.imag,
            gammas,
            betas,
            self.hc_diag,
            self.n_qubits,
            n_trotters,
        )
